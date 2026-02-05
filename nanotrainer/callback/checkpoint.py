import os

import torch

from .base import Callback
from ..trainer.trainer import Trainer


class CheckpointCallback(Callback):

    def __init__(self,
                 save_path: str,
                 save_interval: int = 1000,
                 auto_recover: str = None,
                 recover_path: str = None
                 ):
        """
        Initialize the checkpoint callback.

        Args:
            save_path (str): Path to save checkpoint files.
            save_interval (int): Save a checkpoint every N training steps.
            auto_recover (str): Recover mode when resuming from a checkpoint.
                                - 'resume': load model, optimizer, lr scheduler, scaler, etc.
                                            to continue interrupted training.
                                - 'restart': only load model weights, typically used for fine-tuning or re-training.
            recover_path (str): Path to the checkpoint file used for recovery.
        """
        super().__init__()
        self.save_path = save_path
        self.save_interval = save_interval
        self.auto_recover = auto_recover
        self.recover_path = recover_path

    def on_train_begin(self, trainer: Trainer):
        """
        Perform initialization according to the `auto_recover` setting when training starts.

        Args:
            trainer (Trainer): The core training controller.
        """
        if self.auto_recover is None:
            return

        checkpoint = torch.load(self.recover_path, map_location = trainer.device)
        trainer.model.load_state_dict(checkpoint['model'])

        if self.auto_recover == 'resume':
            trainer.optimizer.load_state_dict(checkpoint['optimizer'])

            if trainer.strategy.lr_scheduler is not None:
                trainer.strategy.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            if trainer.strategy.scaler is not None:
                trainer.strategy.scaler.load_state_dict(checkpoint['scaler'])

            trainer.state.epoch = checkpoint['trainer_state']['epoch']
            trainer.state.global_step = checkpoint['trainer_state']['global_step']

            if trainer.strategy.lr_scheduler is not None:
                trainer.strategy.lr_scheduler.last_epoch = (
                    trainer.state.global_step // trainer.strategy.gradient_accumulation_steps
                )
        elif self.auto_recover == 'restart':
            trainer.state.epoch = 0
            trainer.state.global_step = 0

    def on_step_end(self, trainer):
        """
        Save all training states every `save_interval` steps, including
        the model, optimizer, lr scheduler, scaler, and other related states.

        Args:
            trainer (Trainer): The core training controller.
        """
        if trainer.state.global_step % self.save_interval != 0:
            return

        os.makedirs(self.save_path, exist_ok = True)

        scheduler = trainer.strategy.lr_scheduler
        scaler = trainer.strategy.scaler
        checkpoint = {
            "model": trainer.model.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "lr_scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "trainer_state": {
                "epoch": trainer.state.epoch,
                "global_step": trainer.state.global_step
            }
        }

        path = os.path.join(self.save_path, f'{trainer.state.global_step}.pt')
        torch.save(checkpoint, path)

    def on_train_end(self, trainer):
        """
        Save the model weights at the end of training.

        Args:
            trainer (Trainer): The core training controller.
        """
        checkpoint = {
            "model": trainer.model.state_dict(),
        }
        path = os.path.join(self.save_path, f'{trainer.state.global_step}.pt')
        torch.save(checkpoint, path)