from .base import Callback
from .checkpoint import CheckpointCallback
from .logging import ModelSummaryCallback, LoggingCallback, ExperimentCallback


def get_default_callbacks(save_path: str,
                          save_interval: int = 1000,
                          auto_recover: str = None,
                          recover_path: str = None,
                          log_interval: int = 100
                          ) -> list[Callback]:
    """
    Create and return the default list of training callbacks.

    Args:
        save_path (str): Path to save checkpoint files.
        save_interval (int): Save a checkpoint every N training steps.
        auto_recover (str): Recover mode when resuming from a checkpoint.
                            - 'resume': load model, optimizer, lr scheduler, scaler, etc.
                                        to continue interrupted training.
                            - 'restart': only load model weights, typically used for fine-tuning or re-training.
        recover_path (str): Path to the checkpoint file used for recovery.
        log_interval (int): Log training information every N steps.

    Returns:
        list[Callback]: A list of default callbacks.
    """
    return [
        ExperimentCallback(save_path),
        ModelSummaryCallback(),
        LoggingCallback(log_interval),
        CheckpointCallback(save_path, save_interval, auto_recover, recover_path)
    ]
