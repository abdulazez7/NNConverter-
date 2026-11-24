import torch
from torch import nn
from torch.nn import functional as F

SEQ, VOCAB, WIDTH, HIDDEN, HEADS, CLASSES = 6, 12, 24, 48, 3, 2
EPS = 1e-5


class Linear(nn.Linear):
    def __init__(self, inputs, outputs, binary=False):
        super().__init__(inputs, outputs)
        self.binary = binary

    def forward(self, inputs):
        weight = self.weight
        if self.binary:
            scale = weight.detach().abs().mean(dim=1, keepdim=True)
            effective = scale * torch.where(weight.detach() >= 0, 1.0, -1.0)
            weight = effective + (weight - weight.detach())
        return F.linear(inputs, weight, self.bias)


class InferenceModel(nn.Module):
    def __init__(self, architecture, binary=False):
        super().__init__()
        self.architecture = architecture
        self.token = nn.Embedding(VOCAB, WIDTH)
        self.position = nn.Parameter(torch.randn(SEQ, WIDTH) * 0.15)
        if architecture == "transformer":
            self.q = Linear(WIDTH, WIDTH, binary)
            self.k = Linear(WIDTH, WIDTH, binary)
            self.v = Linear(WIDTH, WIDTH, binary)
            self.proj = Linear(WIDTH, WIDTH, binary)
            self.norm1 = nn.LayerNorm(WIDTH, eps=EPS)
            self.up = Linear(WIDTH, HIDDEN, binary)
            self.down = Linear(HIDDEN, WIDTH, binary)
            self.norm2 = nn.LayerNorm(WIDTH, eps=EPS)
            self.out = Linear(WIDTH, CLASSES, binary)
        elif architecture == "mlp":
            self.up = Linear(SEQ * WIDTH, HIDDEN, binary)
            self.out = Linear(HIDDEN, CLASSES, binary)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

    def forward(self, tokens):
        x = self.token(tokens) + self.position
        if self.architecture == "mlp":
            return self.out(self.up(x.flatten(1)).relu())

        def split_heads(layer):
            return layer(x).reshape(-1, SEQ, HEADS, WIDTH // HEADS).transpose(1, 2)

        q, k, v = split_heads(self.q), split_heads(self.k), split_heads(self.v)
        attention = (q @ k.transpose(-1, -2) / ((WIDTH // HEADS) ** 0.5)).softmax(-1)
        attended = (attention @ v).transpose(1, 2).reshape(-1, SEQ, WIDTH)
        x = self.norm1(x + self.proj(attended))
        x = self.norm2(x + self.down(self.up(x).relu()))
        return self.out(x[:, 0])
