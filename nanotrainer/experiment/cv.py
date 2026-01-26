import torch
import torch.nn as nn

from .config import LeNet5Config


class LeNet5(nn.Module):

    def __init__(self, config: LeNet5Config):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(config.channels, 6, 5, padding = 0),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5, padding = 0),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 120, 5, padding = 0),
            nn.ReLU(inplace = True),
            nn.Flatten(),
            nn.Linear(120, 84),
            nn.ReLU(inplace = True),
            nn.Linear(84, config.num_classes),
        )

    def forward(self, x):
        out = self.net(x)
        return out