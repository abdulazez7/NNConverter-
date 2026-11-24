import torch
from torch import nn


def build():
    torch.manual_seed(7)
    model = nn.Sequential(nn.Linear(9, 17), nn.ReLU(), nn.Linear(17, 4))
    return model, (torch.randn(2, 9),), [(torch.randn(2, 9),) for _ in range(3)]
