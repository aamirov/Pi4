import torch
from torch import nn

from ..layers import CrossBlock


class MetricHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.token = nn.Parameter(torch.empty(1, 1, 1024))
        self.projects_x = nn.Linear(1024, 768)
        self.projects_y = nn.Linear(1024, 768)
        self.blocks = nn.ModuleList(CrossBlock() for _ in range(2))
        self.output = nn.Linear(768, 1)

    def forward(self, tokens, positions, batch_size):
        context = self.projects_y(tokens.reshape(batch_size, -1, 1024).float())
        positions = positions.reshape(batch_size, -1, 2)
        features = self.projects_x(self.token.expand(batch_size, -1, -1))
        for block in self.blocks:
            features = block(features, context, positions)
        return self.output(features).reshape(batch_size).exp()
