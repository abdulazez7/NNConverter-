import torch
from torch import nn


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(7, 5)
        self.b = nn.Linear(3, 5)
        self.head = nn.Linear(10, 2)

    def forward(self, x, y):
        a = self.a(x).tanh()
        b = self.b(y).sigmoid()
        return self.head(torch.cat([a, b], dim=-1)), (a * b).sum(dim=1, keepdim=True)


def build():
    torch.manual_seed(11)
    case = lambda: (torch.randn(3, 7), torch.randn(3, 3))
    return Network(), case(), [case() for _ in range(3)]
