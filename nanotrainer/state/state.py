from dataclasses import dataclass


@dataclass
class TrainState:
    epoch: int = 0
    global_step: int = 0
    loss: float = None
    lr: float = None
