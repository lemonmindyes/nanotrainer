import torch
from enum import Enum

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


class Precision(Enum):
    """
    Enumeration of supported numerical precisions.
    """
    FP32 = 'fp32'
    FP16 = 'fp16'
    BF16 = 'bf16'