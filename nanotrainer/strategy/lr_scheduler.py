import math

import torch
from .base import WarmupSchedulerBase


class WarmupCosineDecay(WarmupSchedulerBase):
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
        super().__init__(optimizer, warmup_ratio, min_lr, last_epoch)

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


class WarmupPolyDecay(WarmupSchedulerBase):
    """
    Polynomial learning rate scheduler with linear warmup.

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
                 power: float = 0.9,
                 min_lr: float = 0.0,
                 last_epoch: int = -1
                 ):
        """
        Args:
            power: Polynomial decay power.
                   Typical values: 0.9 (default), 1.0.
        """
        super().__init__(optimizer, warmup_ratio, min_lr, last_epoch)
        self.power = power

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
        step = self.last_epoch + 1

        # 1. Linear warmup phase
        if step < self.warmup_steps:
            scale = step / max(1, self.warmup_steps)
            return [
                base_lr * scale
                for base_lr in self.base_lrs
            ]

        # 2. Polynomial decay
        progress = (step - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        progress = min(progress, 1.0)
        return [
            self.min_lr +
            (base_lr - self.min_lr) * (1.0 - progress) ** self.power
            for base_lr in self.base_lrs
        ]


class OneCycleDecay(WarmupSchedulerBase):
    """
    One-cycle learning rate scheduler with linear warmup.

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
                 pct_start: float = 0.3,
                 div_factor: float = 25.0,
                 final_div_factor: float = 1e4,
                 last_epoch: int = -1
                 ):
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor

        super().__init__(optimizer, warmup_ratio = pct_start, min_lr = None, last_epoch = last_epoch)

        # injected by Trainer
        self.initial_lrs = None
        self.min_lrs = None

    def lazy_init(self, total_steps: int):
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * self.warmup_ratio)

        self.initial_lrs = [
            lr / self.div_factor for lr in self.base_lrs
        ]
        self.min_lrs = [
            lr / self.final_div_factor for lr in self.base_lrs
        ]

    def get_lr(self):
        # Construction phase: return base learning rates
        if self.total_steps is None:
            return self.base_lrs
        assert self.total_steps is not None, \
            'You must call scheduler.set_total_steps() before training'
        step = self.last_epoch + 1

        # 1. Increase phase: initial -> max
        if step < self.warmup_steps:
            progress = step / max(1, self.warmup_steps)
            return [
                init_lr + 0.5 * (base_lr - init_lr) * (1.0 + math.cos(math.pi * (1.0 - progress)))
                for base_lr, init_lr in zip(self.base_lrs, self.initial_lrs)
            ]

        # 2. Cosine decay phase
        progress = (step - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        progress = min(progress, 1.0)
        return [
            min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
            for base_lr, min_lr in zip(self.base_lrs, self.min_lrs)
        ]