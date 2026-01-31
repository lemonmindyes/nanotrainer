import time
from collections import deque

import matplotlib.pyplot as plt
import yaml

from .base import Callback


class ModelSummaryCallback(Callback):

    def on_train_begin(self, trainer):
        model = trainer.model

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Model Summary')
        print(f'Total parameters:     {total_params / 1e7:.2f}M')
        print(f'Trainable parameters: {trainable_params / 1e7:.2f}M')


class LoggingCallback(Callback):

    def __init__(self, log_interval = 100):
        self.log_interval = log_interval

        self.start_time = None

    def _print(self, trainer):
        e = trainer.state.epoch + 1
        step = trainer.state.global_step
        loss = trainer.state.loss
        lr = trainer.optimizer.param_groups[0]['lr']
        print(f'Epoch: {e}, Step: {step}, Loss: {loss:.2f}, LR: {lr:.6f}, Time: {time.time() - self.start_time:.2f}')

    def on_step_end(self, trainer):
        if trainer.state.global_step % self.log_interval == 0:
            self._print(trainer)
        else:
            return

    def on_train_begin(self, trainer):
        self.start_time = time.time()

    def on_train_end(self, trainer):
        if trainer.state.global_step % self.log_interval != 0:
            self._print(trainer)
        else:
            return


class RealtimePlotCallback(Callback):

    def __init__(self, max_points=1000, log_interval=10):
        super().__init__()
        self.log_interval = log_interval

        self.step_list = deque(maxlen=max_points)
        self.loss_list = deque(maxlen=max_points)
        self.lr_list = deque(maxlen=max_points)

        plt.ion()

        # loss 图
        self.fig_loss, self.ax_loss = plt.subplots()
        self.ax_loss.set_title("Training Loss")

        # lr 图
        self.fig_lr, self.ax_lr = plt.subplots()
        self.ax_lr.set_title("Learning Rate")

    def on_step_end(self, trainer):
        if trainer.state.global_step % self.log_interval != 0:
            return

        self.step_list.append(trainer.state.global_step)
        self.loss_list.append(trainer.state.loss)
        self.lr_list.append(trainer.state.lr)

        # --- loss figure ---
        self.ax_loss.clear()
        self.ax_loss.plot(self.step_list, self.loss_list)
        self.ax_loss.set_xlabel("Step")
        self.ax_loss.set_ylabel("Loss")

        # --- lr figure ---
        self.ax_lr.clear()
        self.ax_lr.plot(self.step_list, self.lr_list)
        self.ax_lr.set_xlabel("Step")
        self.ax_lr.set_ylabel("Learning Rate")

        plt.pause(0.1)


class ExperimentCallback(Callback):

    def __init__(self, save_path: str):
        self.save_path = save_path

    def on_train_begin(self, trainer):
        batch_size = trainer.dataloader.batch_size
        num_workers = trainer.dataloader.num_workers
        pin_memory = trainer.dataloader.pin_memory
        drop_last = trainer.dataloader.drop_last

        model_name = trainer.model.__class__.__name__
        total_params = sum(p.numel() for p in trainer.model.parameters())
        total_trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)

        optimizer = trainer.optimizer.__class__.__name__
        lr = trainer.optimizer.param_groups[0]['lr']
        weight_decay = trainer.optimizer.param_groups[0]['weight_decay']

        strategy = trainer.strategy.__class__.__name__
        device = trainer.strategy.device
        precision = trainer.strategy.precision.value
        use_amp = trainer.strategy.use_amp
        gradient_clip_val = trainer.strategy.gradient_clip_val
        gradient_accumulation_steps = trainer.strategy.gradient_accumulation_steps

        warmup_scheduler = trainer.strategy.lr_scheduler.callback[0].__class__.__name__
        decay_scheduler = trainer.strategy.lr_scheduler.callback[1].__class__.__name__
        warmup_ratio = trainer.strategy.lr_scheduler.warmup_ratio
        min_lr = trainer.strategy.lr_scheduler.min_lr

        callback_names = [callback.__class__.__name__ for callback in trainer.callback]

        recoder = {
            "dataloader": {
                "batch_size": batch_size,
                "num_workers": num_workers,
                "pin_memory": pin_memory,
                "drop_last": drop_last
            },
            "model": {
                "name": model_name,
                "total_params": total_params,
                "total_trainable_params": total_trainable_params,
            },
            "strategy": {
                "name": strategy,
                "device": device,
                "precision": precision,
                "use_amp": use_amp,
                "gradient_clip_val": gradient_clip_val,
                "gradient_accumulation_steps": gradient_accumulation_steps
            },
            "optimizer": {
                "name": optimizer,
                "lr": lr,
                "weight_decay": weight_decay
            },
            "lr_scheduler": {
                "warmup_scheduler": warmup_scheduler,
                "decay_scheduler": decay_scheduler,
                "warmup_ratio": warmup_ratio,
                "min_lr": min_lr
            },
            "callback": callback_names
        }

        with open(f'{self.save_path}/{model_name}.yaml', 'w', encoding = 'utf-8') as f:
            yaml.dump(recoder,
                      f,
                      allow_unicode = True,
                      sort_keys = False,
                      default_flow_style = False,
                      indent = 4
                      )
