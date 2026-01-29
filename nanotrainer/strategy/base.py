import torch
from enum import Enum
from torch.optim.lr_scheduler import LRScheduler

class Strategy:
    """
    Base class for all training strategies.

    A Strategy encapsulates all low-level training behaviors such as:
    - mixed precision (AMP / FP16 / BF16)
    - gradient accumulation
    - gradient clipping
    - optimizer stepping
    - learning rate scheduling
    - distributed logic (DDP, FSDP, etc.)

    The Trainer should only interact with this interface and never
    care about the concrete implementation.
    """

    def autocast_context(self):
        """
        Return a context manager for automatic mixed precision.

        For example:
            - FP32 strategy: returns nullcontext()
            - FP16 / BF16 strategy: returns torch.amp.autocast(...)

        Usage:
            with strategy.autocast_context():
                loss = train_step(...)
        """
        pass

    def backward(self, loss: torch.Tensor):
        """
        Backward propagation logic.

        This method should handle:
            - loss scaling (if using GradScaler)
            - gradient accumulation
            - calling loss.backward() internally

        The Trainer must NOT call loss.backward() directly.
        """
        pass

    def optimizer_step(self, force = False):
        """
        Perform an optimizer step.

        This method should handle:
            - checking accumulation steps
            - gradient clipping
            - scaler.step() and scaler.update() (if AMP)
            - optimizer.step()
            - lr_scheduler.step()
            - optimizer.zero_grad()

        Args:
            force (bool):
                If True, force an optimizer step regardless of
                accumulation state (useful for last batch).
        """
        pass


class WarmupSchedulerBase(LRScheduler):
    """
    Base class for schedulers with warmup + decay.

    This class handles:
        - warmup_ratio
        - lazy_init(total_steps)
        - warmup_steps computation

    Subclasses should
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 warmup_ratio: float = None,
                 min_lr: float = None,
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
        if warmup_ratio is not None:
            assert 0.0 <= warmup_ratio <= 1.0, \
                f'Invalid warmup_ratio = {warmup_ratio}. It must be in the range [0.0, 1.0].'
        if min_lr is not None:
            assert min_lr >= 0.0, \
                f'Invalid min_lr = {min_lr}. It must be non-negative.'
        self.warmup_ratio = warmup_ratio
        self.min_lr = min_lr

        # injected by Trainer
        self.total_steps = None
        self.warmup_steps = None

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


class Precision(Enum):
    """
    Enumeration of supported numerical precisions.
    """
    FP32 = 'fp32'
    FP16 = 'fp16'
    BF16 = 'bf16'