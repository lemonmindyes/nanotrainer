import math

import torch
from torch.optim.lr_scheduler import LRScheduler


class LinearWarmup:

    def __init__(self):
        pass

    def get_lrs(self,
                base_lrs: list[float],
                step: int,
                warmup_steps: int,
                ):
        scale = step / max(1, warmup_steps)
        return [
            lr * scale
            for lr in base_lrs
        ]


class ExpWarmup:

    def __init__(self, exp_ratio = 5.0):
        self.exp_ratio = exp_ratio

    def get_lrs(self,
                base_lrs: list[float],
                step: int,
                warmup_steps: int,
                ):
        x = step / max(1, warmup_steps)
        scale = (math.exp(self.exp_ratio * x) - 1.0) / (math.exp(self.exp_ratio) - 1.0)
        return [
            lr * scale
            for lr in base_lrs
        ]


class LinearDecay:

    def __init__(self):
        pass

    def get_lrs(self,
                base_lrs: list[float],
                min_lr: float,
                step: int,
                warmup_steps: int,
                total_steps: int
                ):
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(progress, 1.0)
        return [
            min_lr + (lr - min_lr) * (1.0 - progress)
            for lr in base_lrs
        ]


class CosineDecay:

    def __init__(self):
        pass

    def get_lrs(self,
                base_lrs: list[float],
                min_lr: float,
                step: int,
                warmup_steps: int,
                total_steps: int
                ):
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(progress, 1.0)
        return [
            min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * progress))
            for lr in base_lrs
        ]


class ComposedLRScheduler(LRScheduler):

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 warmup_ratio: float,
                 min_lr: float = 0.0,
                 last_epoch: int = -1,
                 callback: callable = None,
                 ):
        assert 0.0 <= warmup_ratio < 1.0, \
            f'warmup_ratio must be in [0.0, 1.0)'
        assert min_lr >= 0.0, \
            f'min_lr must be >= 0.0'

        self.warmup_ratio = warmup_ratio
        self.min_lr = min_lr
        if callback is None:
            self.callback = [
                LinearWarmup(),
                LinearDecay(),
            ]
        else:
            self.callback = callback

        # injected by Trainer
        self.total_steps = None
        self.warmup_steps = None

        super().__init__(optimizer, last_epoch)

    def lazy_init(self, total_steps: int):
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * self.warmup_ratio)

    def get_lr(self):
        if self.total_steps is None:
            return self.base_lrs
        step = self.last_epoch + 1  # optimizer step index

        if step < self.warmup_steps:
            return self.callback[0].get_lrs(self.base_lrs, step, self.warmup_steps)
        else:
            return self.callback[1].get_lrs(self.base_lrs, self.min_lr, step, self.warmup_steps, self.total_steps)

