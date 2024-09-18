import os.path

import wandb

from deeper.utils.dist import is_main_process


class WandbWrapper(object):

    def __init__(
            self,
            project,
            name,
            config,
            save_files=None,
            file_prefix=None,
            resume=False,
            save_code=False,
            debug=False
    ):
        if is_main_process() and not debug:
            wandb.init(
                project=project,
                name=name,
                config=config,
                resume=resume,
                save_code=save_code
            )
            if save_files is not None:
                for file in save_files:
                    if file_prefix is not None:
                        file = os.path.join(file_prefix, file)
                    # get base path from file
                    base_path = "/".join(file.split("/")[:-1])
                    wandb.save(file, base_path=base_path)
            self.inited = True
        else:
            self.inited = False

    def log(self, data, step=None):
        if is_main_process() and self.inited:
            wandb.log(data, step)

    def save(self, path, base_path=None):
        if is_main_process() and self.inited:
            wandb.save(path, base_path=base_path)

    def update(self, data):
        if is_main_process() and self.inited:
            wandb.config.update(data, allow_val_change=True)

    def finish(self):
        if is_main_process() and self.inited:
            wandb.finish()
