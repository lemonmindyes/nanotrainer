from .logging import ModelSummaryCallback, LoggingCallback, ExperimentCallback
from .checkpoint import CheckpointCallback


def get_default_callbacks(save_path: str,
                          save_interval: int = 1000,
                          auto_recover: str = None,
                          recover_path: str = None,
                          log_interval: int = 100
                          ):
    return [
        ExperimentCallback(save_path),
        ModelSummaryCallback(),
        LoggingCallback(log_interval),
        CheckpointCallback(save_path, save_interval, auto_recover, recover_path)
    ]
