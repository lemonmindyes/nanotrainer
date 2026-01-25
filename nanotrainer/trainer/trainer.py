import math

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from ..callback.base import Callback
from ..state.state import TrainState
from ..strategy.base import Strategy


class Trainer:
    """
    Core training loop controller.

    Trainer is responsible for:
    - managing the global training time axis (global_step)
    - orchestrating forward / backward / optimizer steps
    - invoking callbacks at proper lifecycle stages

    Note:
        max_steps refers to *forward steps*, not optimizer steps.
        Real optimizer update steps are derived from gradient accumulation.
    """

    def __init__(self,
                 device: torch.device,
                 max_steps: int,
                 *,
                 model: nn.Module,
                 loss_func: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 dataloader: DataLoader,
                 train_step: callable,
                 strategy: Strategy,
                 callback: list[Callback]
                 ):
        """
        Args:
            device: Target device for model and tensors.
            max_steps: Total number of forward steps to run.
                       (Not affected by gradient accumulation.)
            model: Training model.
            loss_func: Loss function.
            optimizer: Optimizer instance.
            dataloader: DataLoader providing training batches.
            train_step: User-defined forward step function.
            strategy: Training strategy (AMP / grad acc / DDP, etc.).
            callback: List of callbacks for logging / saving / monitoring
        """
        # param
        self.device = device
        self.max_steps = max_steps
        # module
        self.model = model.to(device)
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.train_step = train_step
        self.strategy = strategy
        self.callback = callback
        self.state = TrainState()

    def _call_callbacks(self, hook_name: str, *args, **kwargs):
        """
        Invoke a specific callback hook on all registered callbacks.

        Args:
            hook_name: Name of the hook method (e.g. 'on_train_begin')
        """
        for callback in self.callback:
            hook = getattr(callback, hook_name, None)
            if hook is not None:
                hook(*args, **kwargs)

    def fit(self):
        """
        Start training loop.

        This method:
        1. Computes real optimizer step count
        2. Lazily initializes learning rate scheduler.
        3. Runs step-based training until max_steps is reached.
        """
        # Number of real optimizer steps
        optimizer_steps = math.ceil(
            self.max_steps / self.strategy.gradient_accumulation_steps
        )

        # Lazy initialize scheduler with real time axis
        if self.strategy.lr_scheduler is not None:
            self.strategy.lr_scheduler.lazy_init(optimizer_steps)

        self._call_callbacks('on_train_begin', self)

        # Main training loop(step-based)
        while self.state.global_step < self.max_steps:
            for batch in self.dataloader:
                if self.state.global_step >= self.max_steps:
                    break

                self.state.global_step += 1

                # Forward
                with self.strategy.autocast_context():
                    loss = self.train_step(self.model, self.loss_func, batch, self.device)

                self.state.loss = loss.item()
                self._call_callbacks('on_step_end', self)

                # backward & optimizer step
                self.strategy.backward(loss)
                self.strategy.optimizer_step()

                self.state.lr = self.optimizer.param_groups[0]['lr']

            # Epoch is only a derived, display-level concept
            self.state.epoch += 1
            self._call_callbacks('on_epoch_end', self)

        # Flush remaining gradients if using accumulation
        self.strategy.optimizer_step(force = True)
        self._call_callbacks('on_train_end', self)


