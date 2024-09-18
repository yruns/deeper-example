"""
Evaluate Hook

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import torch

from deeper.callbacks.default import CallbackBase
from deeper.utils import comm
from deeper.utils import dist
from tqdm import tqdm


class Evaluator(CallbackBase):
    def on_training_epoch_end(self):
        self.eval()

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        test_loss = 0.0
        correct = 0

        val_iter = enumerate(tqdm(self.trainer.val_loader, desc="Validation")) \
            if dist.is_main_process() else enumerate(self.trainer.val_loader)
        for i, batch in val_iter:
            with torch.no_grad():
                batch = comm.convert_and_move_tensor(batch, self.trainer.mixed_precision, device=self.engine.local_rank)
                data, target = comm.move_tensor_to_device(batch, self.engine.local_rank)
                output, loss = self.trainer.model(data, target)
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum()

            test_loss += loss

        test_loss = dist.all_reduce_average(test_loss) / len(self.trainer.val_loader.dataset)
        acc = dist.all_reduce_sum(correct) / len(self.trainer.val_loader.dataset)

        self.trainer.logger.info(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {acc:.4f}")

        self.trainer.wandb.log({"val_loss": test_loss, "val_acc": acc}, step=self.trainer.completed_steps)

        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        self.trainer.comm_info["current_metric_value"] = acc  # save for saver
        self.trainer.comm_info["current_metric_name"] = "acc"  # save for saver

        if self.trainer.best_metric_value < acc:
            self.trainer.best_metric_value = acc
            self.trainer.best_metric_epoch = self.trainer.epoch
            self.trainer.logger.info("New best metric!")

    def on_training_phase_end(self):
        self.trainer.logger.info(
            "Best {}: {:.4f}, epoch at {:2d}".format("mIoU",
                                                     self.trainer.best_metric_value,
                                                     self.trainer.best_metric_epoch)
        )
