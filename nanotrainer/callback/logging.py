from collections import deque

import matplotlib.pyplot as plt

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

    def __init__(self, log_interval = 10):
        self.log_interval = log_interval

    def _print(self, trainer):
        e = trainer.state.epoch + 1
        step = trainer.state.global_step
        loss = trainer.state.loss
        lr = trainer.optimizer.param_groups[0]['lr']
        print(f'Epoch: {e}, Step: {step}, Loss: {loss:.2f}, LR: {lr:.6f}')

    def on_step_end(self, trainer):
        if trainer.state.global_step % self.log_interval == 0:
            self._print(trainer)
        else:
            return

    def on_train_end(self, trainer):
        if trainer.state.global_step % self.log_interval != 0:
            self._print(trainer)
        else:
            return


class RealtimePlotCallback(Callback):

    def __init__(self, max_points=1000, log_interval=10):
        super().__init__()
        self.log_interval = log_interval
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

        self.loss_list.append(trainer.state.loss)
        self.lr_list.append(trainer.state.lr)

        # --- loss figure ---
        self.ax_loss.clear()
        self.ax_loss.plot(self.loss_list)
        self.ax_loss.set_xlabel("Step")
        self.ax_loss.set_ylabel("Loss")

        # --- lr figure ---
        self.ax_lr.clear()
        self.ax_lr.plot(self.lr_list)
        self.ax_lr.set_xlabel("Step")
        self.ax_lr.set_ylabel("Learning Rate")

        plt.pause(0.001)

