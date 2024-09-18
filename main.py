import argparse
from os import path

import deepspeed
import math
import torch.nn.functional as F
import torch.optim
import torch.optim
import torch.utils.data
import torch.utils.data
import torchvision
import torchvision.transforms as T
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader

from deeper.callbacks.misc import *
from deeper.engine import TrainerBase
from deeper.thirdparty.logging import WandbWrapper
from deeper.thirdparty.logging import logger
from deeper.utils import comm

# os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"

DATASETS_PATH = path.join(path.dirname(__file__), "..", "..", "Datasets")


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)

        self.fc3 = nn.Linear(84, 10)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, labels):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x = self.fc3(x)
        loss = self.criterion(x, labels)
        return x, loss


class Trainer(TrainerBase):

    def __init__(self, hparams, logger, debug=False, callbacks=None):
        super().__init__()
        self.running_loss = None
        self.criterion = None
        self.hparams = hparams
        self.logger = logger
        self.max_epoch = hparams.num_train_epochs
        self.output_dir = hparams.output_dir
        self.callbacks = callbacks or []
        self.debug = debug

        self.ds_config = hparams.ds_config
        self.gradient_accumulation_steps = hparams.gradient_accumulation_steps

    def configure_model(self):
        logger.info("=> creating model ...")
        self.model = Net(self.hparams)
        num_parameters = comm.count_parameters(self.model)
        logger.info(f"Number of parameters: {num_parameters}")

    def configure_dataloader(self):
        # Get the dataset
        transform = T.Compose(
            [T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

        self.train_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
        self.val_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )

        val_sampler = torch.utils.data.distributed.DistributedSampler(
            self.val_dataset, shuffle=False, drop_last=False
        )
        self.val_loader = DataLoader(
            self.val_dataset, shuffle=False, sampler=val_sampler,
            batch_size=self.hparams.per_device_eval_batch_size, num_workers=self.hparams.num_workers
        )

    def setup_post(self):
        self.engine, self.optimizer, self.train_loader, self.lr_scheduler = deepspeed.initialize(
            model=self.model,
            model_parameters=filter(lambda p: p.requires_grad, self.model.parameters()),
            training_data=self.train_dataset,
            config=self.ds_config,
        )

        self.num_update_steps_per_epoch = math.ceil(
            len(self.train_loader) / self.gradient_accumulation_steps)
        self.total_train_steps = self.hparams.num_train_epochs * self.num_update_steps_per_epoch

        if self.engine.bfloat16_enabled():
            self.mixed_precision = "bf16"
        elif self.engine.fp16_enabled():
            self.mixed_precision = "fp16"
        else:
            self.mixed_precision = "fp32"

        self.running_loss = 0.0

    def configure_wandb(self):
        # When debugging, we don't need to log anything.
        self.wandb = WandbWrapper(
            project=self.hparams.log_project,
            name=self.hparams.log_tag,
            config={
                "log_tag": self.hparams.log_tag,
            },
            save_code=False,
            resume=False,
            file_prefix=os.path.join(self.output_dir, "codebase"),
            save_files=[__file__],
            debug=True
        )

    def training_step(self, batch_data, batch_index):
        batch_data = comm.convert_and_move_tensor(batch_data, self.mixed_precision, device=self.engine.local_rank)
        data, target = batch_data

        _, loss = self.engine(data, target)

        self.engine.backward(loss)
        self.engine.step()

        # reduced_loss = dist.all_reduce_average(loss)
        reduced_loss = loss
        # Anything you want to log in terminal
        self.comm_info["terminal_log"] = {"loss": reduced_loss, "data": torch.mean(data)}
        # Anything you want to log in wandb
        self.comm_info["wandb_log"] = {"loss": reduced_loss}


def main(hparams):
    """Main function."""
    deepspeed.init_distributed()

    hparams.save_path = "output/"
    hparams.log_project = "accl_test"
    hparams.log_tag = "init_1"
    hparams.num_workers = 4
    hparams.gradient_accumulation_steps = 1
    hparams.per_device_train_batch_size = 128
    hparams.lr = 1e-3

    hparams.ds_config = {
        "train_batch_size": hparams.per_device_train_batch_size * dist.get_world_size()
                            * hparams.gradient_accumulation_steps,
        "gradient_accumulation_steps": 1,
        "steps_per_print": 2000,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 0.001,
                "betas": [0.8, 0.999],
                "eps": 1e-8,
                "weight_decay": 3e-7,
            },
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 0.001,
                "warmup_num_steps": 1000,
            },
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "bf16": {"enabled": True},
        "fp16": {
            "enabled": False,
            "fp16_master_weights_and_grads": False,
            "loss_scale": 0,
            "loss_scale_window": 500,
            "hysteresis": 2,
            "min_loss_scale": 1,
            "initial_scale_power": 15,
        },
        "fp32_allreduce": True,
        "preserve_fp32": True,
        "wall_clock_breakdown": False,
        "zero_optimization": {
            "stage": 0,
            "allgather_partitions": True,
            "reduce_scatter": True,
            "allgather_bucket_size": 50000000,
            "reduce_bucket_size": 50000000,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "cpu_offload": False,
        },
    }

    comm.seed_everything(hparams.seed)
    # comm.copy_codebase(hparams.save_path)

    logger.add(f"logs/{time.strftime('%Y-%m-%d_%H-%M-%S')}")

    ### Start command:
    ### python -m accelerate.commands.launch --num_processes=2 --main_process_port 10002 tutorials/deeper_accl.py

    from deeper.callbacks.evaluator import Evaluator
    trainer = Trainer(hparams, logger, debug=False, callbacks=[
        Resumer(checkpoint="output/checkpoints/epoch_5"),
        IterationTimer(warmup_iter=2),
        InformationWriter(log_interval=1),
        Evaluator(),
        CheckpointSaver(save_last_only=False),
    ])
    trainer.fit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fabric MNIST Example")
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local rank passed from distributed launcher",
    )
    parser.add_argument("--num_train_epochs", type=int, default=8, metavar="N",
                        help="number of epochs to train (default: 14)")
    parser.add_argument("--gamma", type=float, default=0.7, metavar="M", help="Learning rate step gamma (default: 0.7)")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=64,
        help="Batch size (per device) for the evaluation dataloader.",
    )

    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default="./output", help="Where to store the final model.")
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default="300",
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        # default=None,
        default="output/step_300",
        help="If the training should continue from a checkpoint folder.",
    )
    # Whether to load the best model at the end of training
    parser.add_argument(
        "--load_best_model",
        action="store_true",
        help="Whether to load the best model at the end of training",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"`, and `"dvclive"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )

    hparams = parser.parse_args()
    main(hparams)
