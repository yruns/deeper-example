# Copyright (c) OpenMMLab. All rights reserved.
import ast
import copy
import os
import os.path as osp
import platform
import shutil
import sys
import tempfile
import typing
import uuid
import warnings
from argparse import Action
from importlib import import_module

from addict import Dict

if platform.system() == "Windows":
    import regex as re
else:
    import re

BASE_KEY = "_base_"
DELETE_KEY = "_delete_"
DEPRECATION_KEY = "_deprecation_"
RESERVED_KEYS = ["filename", "text", "pretty_text"]


class ConfigDict(Dict):
    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super(ConfigDict, self).__getattr__(name)
        except KeyError:
            ex = AttributeError(
                f"'{self.__class__.__name__}' object has no " f"attribute '{name}'"
            )
        except Exception as e:
            ex = e
        else:
            return value
        raise ex


def setup_hparams(file_path, options):
    # config name protocol: dataset_name/model_name-exp_name
    hparams = Config.fromfile(file_path)

    if options is not None:
        hparams.merge_from_dict(options)

    return hparams


class Config:
    """A facility for config and config files.

    It supports common file formats as configs: python/json/yaml. The interface
    is the same as a dict object and also allows access config values as
    attributes.

    """

    @staticmethod
    def _validate_py_syntax(filename):
        with open(filename, "r", encoding="utf-8") as f:
            # Setting encoding explicitly to resolve coding issue on windows
            content = f.read()
        try:
            ast.parse(content)
        except SyntaxError as e:
            raise SyntaxError(
                "There are syntax errors in config " f"file {filename}: {e}"
            )

    @staticmethod
    def _substitute_predefined_vars(filename, temp_config_name):
        file_dirname = osp.dirname(filename)
        file_basename = osp.basename(filename)
        file_basename_no_extension = osp.splitext(file_basename)[0]
        file_extname = osp.splitext(filename)[1]
        support_templates = dict(
            fileDirname=file_dirname,
            fileBasename=file_basename,
            fileBasenameNoExtension=file_basename_no_extension,
            fileExtname=file_extname,
        )
        with open(filename, "r", encoding="utf-8") as f:
            # Setting encoding explicitly to resolve coding issue on windows
            config_file = f.read()
        for key, value in support_templates.items():
            regexp = r"\{\{\s*" + str(key) + r"\s*\}\}"
            value = value.replace("\\", "/")
            config_file = re.sub(regexp, value, config_file)
        with open(temp_config_name, "w", encoding="utf-8") as tmp_config_file:
            tmp_config_file.write(config_file)

    @staticmethod
    def _pre_substitute_base_vars(filename, temp_config_name):
        """Substitute base variable placehoders to string, so that parsing
        would work."""
        with open(filename, "r", encoding="utf-8") as f:
            # Setting encoding explicitly to resolve coding issue on windows
            config_file = f.read()
        base_var_dict = {}
        regexp = r"\{\{\s*" + BASE_KEY + r"\.([\w\.]+)\s*\}\}"
        base_vars = set(re.findall(regexp, config_file))
        for base_var in base_vars:
            randstr = f"_{base_var}_{uuid.uuid4().hex.lower()[:6]}"
            base_var_dict[randstr] = base_var
            regexp = r"\{\{\s*" + BASE_KEY + r"\." + base_var + r"\s*\}\}"
            config_file = re.sub(regexp, f'"{randstr}"', config_file)
        with open(temp_config_name, "w", encoding="utf-8") as tmp_config_file:
            tmp_config_file.write(config_file)
        return base_var_dict

    @staticmethod
    def _substitute_base_vars(cfg, base_var_dict, base_cfg):
        """Substitute variable strings to their actual values."""
        cfg = copy.deepcopy(cfg)

        if isinstance(cfg, dict):
            for k, v in cfg.items():
                if isinstance(v, str) and v in base_var_dict:
                    new_v = base_cfg
                    for new_k in base_var_dict[v].split("."):
                        new_v = new_v[new_k]
                    cfg[k] = new_v
                elif isinstance(v, (list, tuple, dict)):
                    cfg[k] = Config._substitute_base_vars(v, base_var_dict, base_cfg)
        elif isinstance(cfg, tuple):
            cfg = tuple(
                Config._substitute_base_vars(c, base_var_dict, base_cfg) for c in cfg
            )
        elif isinstance(cfg, list):
            cfg = [
                Config._substitute_base_vars(c, base_var_dict, base_cfg) for c in cfg
            ]
        elif isinstance(cfg, str) and cfg in base_var_dict:
            new_v = base_cfg
            for new_k in base_var_dict[cfg].split("."):
                new_v = new_v[new_k]
            cfg = new_v

        return cfg

    @staticmethod
    def _file2dict(filename, use_predefined_variables=True):
        filename = osp.abspath(osp.expanduser(filename))
        os.path.exists(filename)
        fileExtname = osp.splitext(filename)[1]
        if fileExtname not in [".py", ".json", ".yaml", ".yml"]:
            raise IOError("Only py/yml/yaml/json type are supported now!")

        with tempfile.TemporaryDirectory() as temp_config_dir:
            temp_config_file = tempfile.NamedTemporaryFile(
                dir=temp_config_dir, suffix=fileExtname
            )
            if platform.system() == "Windows":
                temp_config_file.close()
            temp_config_name = osp.basename(temp_config_file.name)
            # Substitute predefined variables
            if use_predefined_variables:
                Config._substitute_predefined_vars(filename, temp_config_file.name)
            else:
                shutil.copyfile(filename, temp_config_file.name)
            # Substitute base variables from placeholders to strings
            base_var_dict = Config._pre_substitute_base_vars(
                temp_config_file.name, temp_config_file.name
            )

            if filename.endswith(".py"):
                temp_module_name = osp.splitext(temp_config_name)[0]
                sys.path.insert(0, temp_config_dir)
                Config._validate_py_syntax(filename)
                mod = import_module(temp_module_name)
                sys.path.pop(0)

                import types
                cfg_dict = {
                    name: value
                    for name, value in mod.__dict__.items()
                    if not name.startswith("__")
                       and not isinstance(value, types.ModuleType)
                       and not isinstance(value, types.BuiltinFunctionType)
                       and not isinstance(value, types.FunctionType)
                       and not isinstance(value, typing.TypeVar)
                       and not callable(value)
                       and not name.startswith("__")
                }
                # delete imported module
                del sys.modules[temp_module_name]
            elif filename.endswith((".yml", ".yaml", ".json")):
                raise NotImplementedError
            # close temp file
            temp_config_file.close()

        # check deprecation information
        if DEPRECATION_KEY in cfg_dict:
            deprecation_info = cfg_dict.pop(DEPRECATION_KEY)
            warning_msg = (
                f"The config file {filename} will be deprecated " "in the future."
            )
            if "expected" in deprecation_info:
                warning_msg += f' Please use {deprecation_info["expected"]} ' "instead."
            if "reference" in deprecation_info:
                warning_msg += (
                    " More information can be found at "
                    f'{deprecation_info["reference"]}'
                )
            warnings.warn(warning_msg)

        cfg_text = filename + "\n"
        with open(filename, "r", encoding="utf-8") as f:
            # Setting encoding explicitly to resolve coding issue on windows
            cfg_text += f.read()

        if BASE_KEY in cfg_dict:
            cfg_dir = osp.dirname(filename)
            base_filename = cfg_dict.pop(BASE_KEY)
            base_filename = (
                base_filename if isinstance(base_filename, list) else [base_filename]
            )

            cfg_dict_list = list()
            cfg_text_list = list()
            for f in base_filename:
                _cfg_dict, _cfg_text = Config._file2dict(osp.join(cfg_dir, f))
                cfg_dict_list.append(_cfg_dict)
                cfg_text_list.append(_cfg_text)

            base_cfg_dict = dict()
            for c in cfg_dict_list:
                duplicate_keys = base_cfg_dict.keys() & c.keys()
                if len(duplicate_keys) > 0:
                    raise KeyError(
                        "Duplicate key is not allowed among bases. "
                        f"Duplicate keys: {duplicate_keys}"
                    )
                base_cfg_dict.update(c)

            # Substitute base variables from strings to their actual values
            cfg_dict = Config._substitute_base_vars(
                cfg_dict, base_var_dict, base_cfg_dict
            )

            base_cfg_dict = Config._merge_a_into_b(cfg_dict, base_cfg_dict)
            cfg_dict = base_cfg_dict

            # merge cfg_text
            cfg_text_list.append(cfg_text)
            cfg_text = "\n".join(cfg_text_list)

        return cfg_dict, cfg_text

    @staticmethod
    def _merge_a_into_b(a, b, allow_list_keys=False):
        """merge dict ``a`` into dict ``b`` (non-inplace).

        Values in ``a`` will overwrite ``b``. ``b`` is copied first to avoid
        in-place modifications.

        Args:
            a (dict): The source dict to be merged into ``b``.
            b (dict): The origin dict to be fetch keys from ``a``.
            allow_list_keys (bool): If True, int string keys (e.g. '0', '1')
              are allowed in source ``a`` and will replace the element of the
              corresponding index in b if b is a list. Default: False.

        Returns:
            dict: The modified dict of ``b`` using ``a``.

        """
        b = b.copy()
        for k, v in a.items():
            if allow_list_keys and k.isdigit() and isinstance(b, list):
                k = int(k)
                if len(b) <= k:
                    raise KeyError(f"Index {k} exceeds the length of list {b}")
                b[k] = Config._merge_a_into_b(v, b[k], allow_list_keys)
            elif isinstance(v, dict) and k in b and not v.pop(DELETE_KEY, False):
                allowed_types = (dict, list) if allow_list_keys else dict
                if not isinstance(b[k], allowed_types):
                    raise TypeError(
                        f"{k}={v} in child config cannot inherit from base "
                        f"because {k} is a dict in the child config but is of "
                        f"type {type(b[k])} in base config. You may set "
                        f"`{DELETE_KEY}=True` to ignore the base config"
                    )
                b[k] = Config._merge_a_into_b(v, b[k], allow_list_keys)
            else:
                b[k] = v
        return b

    @staticmethod
    def fromfile(filename, use_predefined_variables=True, import_custom_modules=True):
        cfg_dict, cfg_text = Config._file2dict(filename, use_predefined_variables)
        # if import_custom_modules and cfg_dict.get("custom_imports", None):
        #     import_modules_from_strings(**cfg_dict["custom_imports"])
        return Config(cfg_dict, cfg_text=cfg_text, filename=filename)

    @staticmethod
    def fromstring(cfg_str, file_format):
        """Generate config from config str.

        Args:
            cfg_str (str): Config str.
            file_format (str): Config file format corresponding to the
               config str. Only py/yml/yaml/json type are supported now!

        Returns:
            obj:`Config`: Config obj.
        """
        if file_format not in [".py", ".json", ".yaml", ".yml"]:
            raise IOError("Only py/yml/yaml/json type are supported now!")
        if file_format != ".py" and "dict(" in cfg_str:
            # check if users specify a wrong suffix for python
            warnings.warn('Please check "file_format", the file format may be .py')
        with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", suffix=file_format, delete=False
        ) as temp_file:
            temp_file.write(cfg_str)
            # on windows, previous implementation cause error
            # see PR 1077 for details
        cfg = Config.fromfile(temp_file.name)
        os.remove(temp_file.name)
        return cfg

    def __init__(self, cfg_dict=None, cfg_text=None, filename=None):
        if cfg_dict is None:
            cfg_dict = dict()
        elif not isinstance(cfg_dict, dict):
            raise TypeError("cfg_dict must be a dict, but " f"got {type(cfg_dict)}")
        for key in cfg_dict:
            if key in RESERVED_KEYS:
                raise KeyError(f"{key} is reserved for config file")

        super(Config, self).__setattr__("_cfg_dict", ConfigDict(cfg_dict))
        super(Config, self).__setattr__("_filename", filename)
        if cfg_text:
            text = cfg_text
        elif filename:
            with open(filename, "r") as f:
                text = f.read()
        else:
            text = ""
        super(Config, self).__setattr__("_text", text)

    @property
    def filename(self):
        return self._filename

    @property
    def text(self):
        return self._text

    def try_get(self, name):
        if hasattr(self._cfg_dict, name):
            return getattr(self._cfg_dict, name)
        return None

    def __repr__(self):
        return f"Config (path: {self.filename}): {self._cfg_dict.__repr__()}"

    def __len__(self):
        return len(self._cfg_dict)

    def __getattr__(self, name):
        return getattr(self._cfg_dict, name)

    def __getitem__(self, name):
        return self._cfg_dict.__getitem__(name)

    def __setattr__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setattr__(name, value)

    def __setitem__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setitem__(name, value)

    def __iter__(self):
        return iter(self._cfg_dict)

    def __getstate__(self):
        return (self._cfg_dict, self._filename, self._text)

    def __setstate__(self, state):
        _cfg_dict, _filename, _text = state
        super(Config, self).__setattr__("_cfg_dict", _cfg_dict)
        super(Config, self).__setattr__("_filename", _filename)
        super(Config, self).__setattr__("_text", _text)

    def merge_from_dict(self, options, allow_list_keys=True):
        """Merge list into cfg_dict.

        Merge the dict parsed by MultipleKVAction into this cfg.

        Args:
            options (dict): dict of configs to merge from.
            allow_list_keys (bool): If True, int string keys (e.g. '0', '1')
              are allowed in ``options`` and will replace the element of the
              corresponding index in the config if the config is a list.
              Default: True.
        """
        option_cfg_dict = {}
        for full_key, v in options.items():
            d = option_cfg_dict
            key_list = full_key.split(".")
            for subkey in key_list[:-1]:
                d.setdefault(subkey, ConfigDict())
                d = d[subkey]
            subkey = key_list[-1]
            d[subkey] = v

        cfg_dict = super(Config, self).__getattribute__("_cfg_dict")
        super(Config, self).__setattr__(
            "_cfg_dict",
            Config._merge_a_into_b(
                option_cfg_dict, cfg_dict, allow_list_keys=allow_list_keys
            ),
        )


class DictAction(Action):
    """
    argparse action to split an argument into KEY=VALUE form
    on the first = and append to a dictionary. List options can
    be passed as comma separated values, i.e 'KEY=V1,V2,V3', or with explicit
    brackets, i.e. 'KEY=[V1,V2,V3]'. It also support nested brackets to build
    list/tuple values. e.g. 'KEY=[(V1,V2),(V3,V4)]'
    """

    @staticmethod
    def _parse_int_float_bool(val):
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            pass
        if val.lower() in ["true", "false"]:
            return True if val.lower() == "true" else False
        return val

    @staticmethod
    def _parse_iterable(val):
        """Parse iterable values in the string.

        All elements inside '()' or '[]' are treated as iterable values.

        Args:
            val (str): Value string.

        Returns:
            list | tuple: The expanded list or tuple from the string.

        Examples:
            >>> DictAction._parse_iterable('1,2,3')
            [1, 2, 3]
            >>> DictAction._parse_iterable('[a, b, c]')
            ['a', 'b', 'c']
            >>> DictAction._parse_iterable('[(1, 2, 3), [a, b], c]')
            [(1, 2, 3), ['a', 'b'], 'c']
        """

        def find_next_comma(string):
            """Find the position of next comma in the string.

            If no ',' is found in the string, return the string length. All
            chars inside '()' and '[]' are treated as one element and thus ','
            inside these brackets are ignored.
            """
            assert (string.count("(") == string.count(")")) and (
                    string.count("[") == string.count("]")
            ), f"Imbalanced brackets exist in {string}"
            end = len(string)
            for idx, char in enumerate(string):
                pre = string[:idx]
                # The string before this ',' is balanced
                if (
                        (char == ",")
                        and (pre.count("(") == pre.count(")"))
                        and (pre.count("[") == pre.count("]"))
                ):
                    end = idx
                    break
            return end

        # Strip ' and " characters and replace whitespace.
        val = val.strip("'\"").replace(" ", "")
        is_tuple = False
        if val.startswith("(") and val.endswith(")"):
            is_tuple = True
            val = val[1:-1]
        elif val.startswith("[") and val.endswith("]"):
            val = val[1:-1]
        elif "," not in val:
            # val is a single value
            return DictAction._parse_int_float_bool(val)

        values = []
        while len(val) > 0:
            comma_idx = find_next_comma(val)
            element = DictAction._parse_iterable(val[:comma_idx])
            values.append(element)
            val = val[comma_idx + 1:]
        if is_tuple:
            values = tuple(values)
        return values

    def __call__(self, parser, namespace, values, option_string=None):
        options = {}
        for kv in values:
            key, val = kv.split("=", maxsplit=1)
            options[key] = self._parse_iterable(val)
        setattr(namespace, self.dest, options)
