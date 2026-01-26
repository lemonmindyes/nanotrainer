from dataclasses import dataclass

@dataclass
class LeNet5Config:
    input_size: int = 32
    channels: int = 3
    num_classes: int = 10