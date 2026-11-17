import math

import torch
from torch import nn

from .layers import Block


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, 1024, 16, stride=16)
        self.prefix_tokens = nn.Parameter(torch.empty(1, 5, 1024))
        self.blocks = nn.ModuleList(
            Block(1024, 16, epsilon=1e-5, scaled=True) for _ in range(18)
        )
        self.norm = nn.LayerNorm(1024, eps=1e-5)
        self.projection = nn.Linear(2048, 1024)
        self.register_buffer("periods", 100 ** (torch.arange(16).float() / 16))

    def forward(self, images):
        features = self.patch_embed(images)
        batch_size, _, height, width = features.shape
        features = torch.cat(
            (
                self.prefix_tokens.expand(batch_size, -1, -1),
                features.flatten(2).transpose(1, 2),
            ),
            dim=1,
        )
        coordinates = torch.meshgrid(
            torch.arange(height, device=images.device).float() + 0.5,
            torch.arange(width, device=images.device).float() + 0.5,
            indexing="ij",
        )
        coordinates = (
            torch.stack(coordinates, dim=-1).flatten(0, 1) / max(height, width) * 2 - 1
        )
        angles = (
            (2 * math.pi * coordinates[:, :, None] / self.periods.float())
            .flatten(1)
            .repeat(1, 2)
        )
        rotary = angles.sin(), angles.cos()
        intermediate = []
        for index, block in enumerate(self.blocks):
            features = block(features, rotary=rotary)
            if index in (11, 17):
                intermediate.append(self.norm(features)[:, 5:])
        return self.projection(torch.cat(intermediate, dim=-1))
