"""
Default engine/testing logic
modified from Pointcept(https://github.com/Pointcept/Pointcept)

Please cite our work if the code is helpful to you.
"""

import shutil
from os.path import join

import torch

import deeper.utils.comm as comm


def worker_init_fn(worker_id, num_workers, rank, seed):
    """Worker init func for dataloader.

    The seed of each worker equals to num_worker * rank + worker_id + user_seed

    Args:
        worker_id (int): Worker id.
        num_workers (int): Number of workers.
        rank (int): The rank of current process.
        seed (int): The random seed to use.
    """

    worker_seed = num_workers * rank + worker_id + seed
    comm.seed_everything(worker_seed)


def save_checkpoint(state, is_best, save_path, filename='model_last.pth.tar'):
    filename = join(save_path, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, join(save_path, 'model_best.pth.tar'))


def save_checkpoint_epoch(state, save_path, epoch):
    filename = join(save_path, f'model_epoch_{epoch}.pth.tar')
    torch.save(state, filename)


def load_state_dict(state_dict, model, logger, strict=False):
    try:
        load_state_info = model.load_state_dict(state_dict, strict=strict)
    except Exception:
        # The model was trained in a parallel manner, so need to be loaded differently
        from collections import OrderedDict
        weight = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                # remove module
                k = k[7:]  # module.xxx.xxx -> xxx.xxx
            else:
                # add module
                k = 'module.' + k  # xxx.xxx -> module.xxx.xxx
            weight[k] = v
        load_state_info = model.load_state_dict(weight, strict=strict)
    if not strict:
        logger.info(f"Missing keys: {load_state_info.missing_keys}")
        logger.info(f"Unexpected keys: {load_state_info.unexpected_keys}")

    return model
