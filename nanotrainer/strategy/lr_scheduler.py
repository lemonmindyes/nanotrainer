import math

import torch
from torch.optim.lr_scheduler import LRScheduler


class CosineWarmupDecay(LRScheduler):
    """
    Cosine learning rate scheduler with linear warmup.

    This scheduler operates on *optimizer steps*, not forward steps.
    The real training time axis must be injected via `lazy_init`
    before the first optimizer update.

    Typical lifecycle:
        1. Construct scheduler.
        2. Trainer computes real optimizer step count.
        3. Trainer calls `lazy_init(total_steps)`.
        4. Strategy calls `scheduler.step()` after each optimizer.step().
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 warmup_ratio: float,
                 min_lr: float = 0.0,
                 last_epoch: int = -1
                 ):
        """
        Args:
            optimizer: Wrapped optimizer.
            warmup_ratio: Ratio of warmup steps relative to total optimizer steps.
                          Value should be in (0, 1).
            min_lr: Minimum learning rate at the end of decay.
            last_epoch: Index of last optimizer step.
                        Should remain -1 for fresh training./
        """
        self.warmup_ratio = warmup_ratio
        self.min_lr = min_lr

        # lazy init by Trainer
        self.total_steps = None # total optimizer update steps
        self.warmup_steps = None # warmup optimizer steps

        super().__init__(optimizer, last_epoch)

    def lazy_init(self, total_steps: int):
        """
        Inject real training time axis

        This method must be called before training starts.

        Args:
            total_steps: Total number of optimizer update steps
        """
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * self.warmup_ratio)

    def get_lr(self):
        """
        Compute learning rate for current optimizer step.

        Note:
            During scheduler construction, Pytorch will trigger
            an initial step. In this case, time axis is not ready
            and base learning rates are returned.
        """
        # Construction phase: return base learning rates
        if self.total_steps is None:
            return self.base_lrs
        assert self.total_steps is not None, \
            'You must call scheduler.set_total_steps() before training'
        step = self.last_epoch + 1 # optimizer step index

        # 1. Linear warmup phase
        if step < self.warmup_steps:
            scale = step / max(1, self.warmup_steps)
            return [
                base_lr * scale
                for base_lr in self.base_lrs
            ]

        # 2. Cosine decay phase
        progress = (step - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        progress = min(progress, 1.0)
        return [
            self.min_lr + 0.5 * (base_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress))
            for base_lr in self.base_lrs
        ]
