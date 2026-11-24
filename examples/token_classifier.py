import torch
from torch import nn


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokens = nn.Embedding(32, 10)
        self.norm = nn.LayerNorm(10)
        self.head = nn.Linear(10, 4)

    def forward(self, tokens):
        return self.head(self.norm(self.tokens(tokens)).mean(dim=1))


def build():
    torch.manual_seed(10)
    return (
        Network(),
        (torch.randint(0, 32, (2, 6)),),
        [(torch.randint(0, 32, (2, 6)),) for _ in range(3)],
    )
