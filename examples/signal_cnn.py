import torch
from torch import nn


def build():
    torch.manual_seed(12)
    model = nn.Sequential(
        nn.Conv1d(2, 4, 3, padding=1, dilation=1),
        nn.LeakyReLU(0.2),
        nn.AvgPool1d(2),
        nn.Flatten(),
        nn.Linear(32, 3),
    )
    return model, (torch.randn(2, 2, 16),), [(torch.randn(2, 2, 16),) for _ in range(3)]
