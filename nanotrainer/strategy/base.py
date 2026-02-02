import torch
from enum import Enum
from torch.optim.lr_scheduler import LRScheduler

class Strategy:
    """
    Base class for all training strategies.
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
        Backward gradient propagation.

        This method should handle:
            - loss scaling (if using GradScaler)
            - gradient accumulation
            - calling loss.backward() internally

        Args:
            loss (torch.Tensor):
                Scalar loss tensor produced by the current forward pass.

        The Trainer must NOT call loss.backward() directly.
        """
        pass

    def optimizer_step(self, force: bool = False):
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


class Precision(Enum):
    """
    Enumeration of supported numerical precisions. (FP32,FP16,BF16)
    """
    FP32 = 'fp32'
    FP16 = 'fp16'
    BF16 = 'bf16'