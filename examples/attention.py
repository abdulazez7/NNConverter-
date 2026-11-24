import torch
from torch import nn


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(12, 12)
        self.k = nn.Linear(12, 12)
        self.v = nn.Linear(12, 12)
        self.norm = nn.LayerNorm(12)
        self.ff = nn.Sequential(nn.Linear(12, 20), nn.GELU(), nn.Linear(20, 12))
        self.head = nn.Linear(12, 3)

    def forward(self, x):
        z = (self.q(x) @ self.k(x).transpose(-2, -1) / (12**0.5)).softmax(
            -1
        ) @ self.v(x)
        z = self.norm(x + z)
        return self.head((z + self.ff(z)).mean(dim=1))


def build():
    torch.manual_seed(9)
    return (
        Network(),
        (torch.randn(2, 5, 12),),
        [(torch.randn(2, 5, 12),) for _ in range(3)],
    )
