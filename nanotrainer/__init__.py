VERSION = "0.0.1"
__author__ = "lemonmindyes"
__license__ = "Apache-2.0"
__description__ = "A naon experimental trainer for Pytorch"

# callback
from .callback import checkpoint, default, logging

# experiment
from .experiment import config, cv

# state

# strategy
from .strategy import lr_scheduler, single

# trainer
from .trainer import trainer


__all__ = [
    "logging",
    "checkpoint",
    "lr_scheduler",
    "single",
    "trainer"
]
