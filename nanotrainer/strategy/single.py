from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

from .base import Strategy, Precision
from ..trainer.trainer import Trainer


class SingleStrategy(Strategy):
    """
    Training strategy for single-device execution.

    This strategy encapsulates all low-level training mechanics for
    single-GPU or single-CPU scenarios, including:

        - Automatic Mixed Precision (AMP)
        - Gradient accumulation
        - Gradient clipping
        - Optimizer stepping
        - Learning rate scheduling

    It acts as a complete training backend.
    The Trainer should treat this class as a black box and never
    perform any low-level training operations by itself.
    """
    
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 device: str = 'cuda',
                 precision: Precision = Precision.BF16,
                 use_amp: bool = True,
                 gradient_clip_val: float = 1.0,
                 lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
                 gradient_accumulation_steps: int = 1
                 ):
        """
        Args:
            model:
                Neural network model.

            optimizer:
                Optimizer instance bound to model parameters.

            device:
                Device type string used by autocast.
                Typical values: "cuda", "cpu".

            precision:
                Numerical precision mode.
                - FP32: no mixed precision
                - FP16: half precision (uses GradScaler)
                - BF16: bfloat16 (no overflow, but still uses autocast)

            use_amp:
                Whether to enable automatic mixed precision.
                Only effective when precision is FP16 or BF16.

            gradient_clip_val:
                Maximum L2 norm for gradient clipping.
                Set <= 0 to disable gradient clipping.

            lr_scheduler:
                Optional learning rate scheduler.
                Will be stepped once after every optimizer step.

            gradient_accumulation_steps:
                Number of backward passes to accumulate gradients
                before performing one optimizer update.

                Example:
                    gradient_accumulation_steps = 4
                    -> optimizer.step() is called every 4 batches.
        """
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.precision = precision
        self.gradient_clip_val = gradient_clip_val
        self.lr_scheduler = lr_scheduler
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # Internal counter for gradient accumulation
        self._accumulation_step = 0

        # Enable AMP only when using FP16 or BF16
        self.use_amp = precision in [Precision.BF16, Precision.FP16] and use_amp
        self.scaler = GradScaler() if self.use_amp else None

        # Determine autocast dtype
        if precision == Precision.BF16:
            self.amp_dtype = torch.bfloat16
        elif precision == Precision.FP16:
            self.amp_dtype = torch.float16
        else:
            self.amp_dtype = None

    def autocast_context(self):
        """
        Return an autocast context manager.

        Behavior:
            - If AMP is disabled: returns a no-op context.
            - If AMP is enabled: returns torch.amp.autocast with
              the configured device type and dtype.

        Typical usage:
            with strategy.autocast_context():
                loss = train_step(...)
        """
        if not self.use_amp:
            return nullcontext()
        return autocast(device_type = self.device, dtype = self.amp_dtype)

    def backward(self, loss):
        """
        Perform backward propagation on the given loss.

        This method handles:
            - Loss normalization for gradient accumulation.
            - AMP scaling (if enabled).
            - Internal accumulation step tracking.

        Args:
            loss:
                Scalar loss tensor produced by the forward pass.
                It will be automatically divided by
                gradient_accumulation_steps.
        """
        # Normalize loss so that accumulated gradient magnitude
        # matches non-accumulated training
        loss = loss / self.gradient_accumulation_steps

        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Record one backward call
        self._accumulation_step += 1

    def optimizer_step(self, force = False):
        """
        Perform an optimizer step if accumulation condition is met.

        Behavior:
            - If force = False:
                Only step when accumulated steps reach
                gradient_accumulation_steps.
            - If force = True:
                Force stepping even if remaining steps are fewer
                than gradient_accumulation_steps.

        The force mode is typically used at the very end of training
        to flush remaining gradients.

        Args:
            force:
                Whether to ignore accumulation threshold and
                force an optimizer update.
        """

        # Nothing to flush
        if force and self._accumulation_step == 0:
            return

        # Not enough accumulated steps yet
        if not force and self._accumulation_step % self.gradient_accumulation_steps != 0:
            return

        # ---- AMP path (FP16 / BF16) ----
        if self.scaler is not None:
            # Unscale gradients for correct clipping
            self.scaler.unscale_(self.optimizer)

            if self.gradient_clip_val > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)

            prev_scale = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # If scale was reduced, optimizer step was skipped
            if self.scaler.get_scale() < prev_scale:
                self.optimizer.zero_grad(set_to_none = True)
                self._accumulation_step = 0
                return

        # ---- FP32 path ----
        else:
            if self.gradient_clip_val > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
            self.optimizer.step()

        # Step learning rate scheduler after successful update
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        # Clear gradients and reset accumulation counter
        self.optimizer.zero_grad(set_to_none = True)
        self._accumulation_step = 0