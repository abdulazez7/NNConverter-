import torch
from torch import nn


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 6, 3, padding=1)
        self.depthwise = nn.Conv2d(6, 6, 3, padding=1, groups=6)
        self.norm = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2)
        self.head = nn.Linear(6, 4)

    def forward(self, x):
        x = self.pool(self.conv(x).relu())
        x = (self.norm(self.depthwise(x)) + x).relu()
        return self.head(x.mean(dim=(2, 3)))


def build():
    torch.manual_seed(8)
    return (
        Network(),
        (torch.randn(1, 3, 8, 8),),
        [(torch.randn(1, 3, 8, 8),) for _ in range(3)],
    )
