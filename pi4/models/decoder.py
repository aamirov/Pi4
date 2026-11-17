import torch
from torch import nn

from .layers import Block


class TimeGates(nn.Module):
    def __init__(self, steps):
        super().__init__()
        self.register_buffer("encoding", torch.empty(steps, 128))
        self.proj = nn.Sequential(nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 3072))

    def forward(self, step, batch_size):
        encoding = self.encoding[step].expand(batch_size, -1).contiguous()
        gates = self.proj(encoding) + 1
        return gates.reshape(batch_size, 3, 1024).permute(1, 0, 2).unsqueeze(2)


class GeometryDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_tokens = nn.Parameter(torch.empty(1, 1, 5, 1024))
        self.frame_block = Block(1024, 16, normalize=True, scaled=True)
        self.global_block = Block(1024, 16, normalize=True, scaled=True)
        self.readout = Block(1024, 16, normalize=True, scaled=True)
        self.frame_gates = TimeGates(16)
        self.global_gates = TimeGates(16)

    def forward(self, tokens, positions, batch_size, frames):
        initial = torch.cat(
            (
                self.register_tokens.expand(batch_size, frames, -1, -1).reshape(
                    batch_size * frames, 5, 1024
                ),
                tokens,
            ),
            dim=1,
        )
        features = initial
        token_count = features.shape[1]
        for index in range(16):
            features = self.frame_block(
                features.reshape(batch_size * frames, token_count, 1024),
                positions,
                gates=self.frame_gates(index, batch_size * frames),
            )
            features = self.global_block(
                features.reshape(batch_size, frames * token_count, 1024),
                gates=self.global_gates(index, batch_size),
            )
        features = self.readout(
            features.reshape(batch_size * frames, token_count, 1024), positions
        )
        return features, initial


class SpatialCondition(nn.Module):
    def __init__(self):
        super().__init__()
        self.geometry = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, 256))
        self.time = nn.Linear(64, 256)
        self.output = nn.Sequential(nn.SiLU(), nn.Linear(256, 4096))
        self.register_buffer("time_encoding", torch.empty(8, 64))

    def forward(self, geometry, step):
        encoding = self.time_encoding[step].expand(geometry.shape[0], -1).contiguous()
        return self.output(
            self.geometry(geometry.float()) + self.time(encoding)[:, None]
        ).chunk(4, dim=-1)


class MotionDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.frame_block = Block(1024, 16, normalize=True, scaled=True)
        self.global_block = Block(1024, 16, normalize=True, scaled=True)
        self.readout = Block(1024, 16, normalize=True, scaled=True)
        self.frame_condition = SpatialCondition()
        self.global_condition = SpatialCondition()
        self.frame_gates = TimeGates(8)
        self.global_gates = TimeGates(8)

    def forward(self, initial, geometry, positions, batch_size, frames):
        token_count = initial.shape[1]
        features = initial
        for index in range(8):
            frame_geometry = geometry.reshape(batch_size * frames, token_count, 1024)
            features = self.frame_block(
                features.reshape_as(frame_geometry),
                positions,
                gates=self.frame_gates(index, batch_size * frames),
                modulation=self.frame_condition(frame_geometry, index),
            )
            global_geometry = geometry.reshape(batch_size, frames * token_count, 1024)
            features = self.global_block(
                features.reshape_as(global_geometry),
                gates=self.global_gates(index, batch_size),
                modulation=self.global_condition(global_geometry, index),
            )
        return self.readout(
            features.reshape(batch_size * frames, token_count, 1024), positions
        ).float()
