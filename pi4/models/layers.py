import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        hidden_dimension = 4 * dimension
        self.fc1 = nn.Linear(dimension, hidden_dimension)
        self.fc2 = nn.Linear(hidden_dimension, dimension)

    def forward(self, features):
        return self.fc2(F.gelu(self.fc1(features)))


def rotate_half(features):
    first, second = features.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def grid_rope(features, positions):
    dimension = features.shape[-1] // 2
    frequencies = 1.0 / (
        100.0
        ** (torch.arange(0, dimension, 2, device=features.device).float() / dimension)
    )
    indices = torch.arange(positions.shape[1] + 1, device=features.device).float()
    angles = torch.einsum("i,j->ij", indices, frequencies).to(features.dtype)
    angles = torch.cat((angles, angles), dim=-1)
    sine, cosine = angles.sin(), angles.cos()
    output = []
    for axis, tokens in enumerate(features.chunk(2, dim=-1)):
        axis_cosine = F.embedding(positions[..., axis], cosine)[:, None]
        axis_sine = F.embedding(positions[..., axis], sine)[:, None]
        output.append(tokens * axis_cosine + rotate_half(tokens) * axis_sine)
    return torch.cat(output, dim=-1)


def patch_positions(batch_size, height, width, device):
    rows, columns = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    positions = torch.stack((rows, columns), dim=-1).reshape(1, -1, 2) + 1
    prefix = torch.zeros(1, 5, 2, device=device, dtype=positions.dtype)
    return torch.cat((prefix, positions), dim=1).expand(batch_size, -1, -1)


class Attention(nn.Module):
    def __init__(self, dimension, heads, normalize=False):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dimension, 3 * dimension)
        self.proj = nn.Linear(dimension, dimension)
        self.q_norm = nn.LayerNorm(dimension // heads) if normalize else nn.Identity()
        self.k_norm = nn.LayerNorm(dimension // heads) if normalize else nn.Identity()

    def forward(self, features, positions=None, rotary=None):
        batch_size, token_count, dimension = features.shape
        projected = self.qkv(features).reshape(
            batch_size, token_count, 3, self.heads, dimension // self.heads
        )
        query, key, value = projected.permute(2, 0, 3, 1, 4).unbind(0)
        query, key = self.q_norm(query).to(value.dtype), self.k_norm(key).to(
            value.dtype
        )
        if positions is not None:
            query, key = grid_rope(query, positions), grid_rope(key, positions)
        if rotary is not None:
            sine, cosine = rotary
            prefix = token_count - sine.shape[0]

            def apply_rotary(tokens):
                spatial = tokens[:, :, prefix:].float()
                spatial = spatial * cosine + rotate_half(spatial) * sine
                return torch.cat(
                    (tokens[:, :, :prefix], spatial.to(tokens.dtype)), dim=2
                )

            query, key = apply_rotary(query), apply_rotary(key)
        attended = F.scaled_dot_product_attention(query, key, value)
        return self.proj(
            attended.transpose(1, 2).reshape(batch_size, token_count, dimension)
        )


class Block(nn.Module):
    def __init__(self, dimension, heads, normalize=False, epsilon=1e-6, scaled=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dimension, eps=epsilon)
        self.attn = Attention(dimension, heads, normalize)
        self.norm2 = nn.LayerNorm(dimension, eps=epsilon)
        self.mlp = MLP(dimension)
        self.residual_scales = (
            nn.Parameter(torch.ones(2, dimension)) if scaled else None
        )

    def forward(
        self, features, positions=None, rotary=None, gates=None, modulation=None
    ):
        attention_scale = feedforward_scale = output_scale = 1
        if gates is not None:
            attention_scale, feedforward_scale, output_scale = gates.to(
                features.dtype
            ).unbind(0)
        attention_input = self.norm1(features)
        if modulation is not None:
            attention_input = (1 + modulation[0]) * attention_input + modulation[1]
        attention_output = self.attn(attention_input, positions, rotary)
        if self.residual_scales is not None:
            attention_output = attention_output * self.residual_scales[0]
        features = features + attention_scale * attention_output
        feedforward_input = self.norm2(features)
        if modulation is not None:
            feedforward_input = (1 + modulation[2]) * feedforward_input + modulation[3]
        feedforward_output = self.mlp(feedforward_input)
        if self.residual_scales is not None:
            feedforward_output = feedforward_output * self.residual_scales[1]
        return output_scale * (features + feedforward_scale * feedforward_output)


class CrossAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(768, 768)
        self.k_proj = nn.Linear(768, 768)
        self.v_proj = nn.Linear(768, 768)
        self.proj = nn.Linear(768, 768)

    def forward(self, query_features, context, query_positions, context_positions):
        batch_size, query_count, _ = query_features.shape
        query = (
            self.q_proj(query_features)
            .reshape(batch_size, query_count, 12, 64)
            .transpose(1, 2)
        )
        key = self.k_proj(context).reshape(batch_size, -1, 12, 64).transpose(1, 2)
        value = self.v_proj(context).reshape(batch_size, -1, 12, 64).transpose(1, 2)
        query, key = grid_rope(query, query_positions), grid_rope(
            key, context_positions
        )
        attended = F.scaled_dot_product_attention(query, key, value)
        return self.proj(attended.transpose(1, 2).reshape(batch_size, query_count, 768))


class CrossBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm2 = nn.LayerNorm(768, eps=1e-6)
        self.norm_y = nn.LayerNorm(768, eps=1e-6)
        self.cross_attn = CrossAttention()
        self.norm3 = nn.LayerNorm(768, eps=1e-6)
        self.mlp = MLP(768)

    def forward(self, features, context, positions):
        features = features + self.cross_attn(
            self.norm2(features), self.norm_y(context), positions[:, :1], positions
        )
        return features + self.mlp(self.norm3(features))


class ConvResidual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.GroupNorm(1, channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode="replicate"),
            nn.GroupNorm(channels // 32, channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode="replicate"),
        )

    def forward(self, features):
        return features + self.layers(features)


def upsample_block(input_channels, output_channels):
    return nn.Sequential(
        nn.Conv2d(
            input_channels, output_channels * 4, 3, padding=1, padding_mode="replicate"
        ),
        nn.PixelShuffle(2),
        nn.Conv2d(
            output_channels, output_channels, 3, padding=1, padding_mode="replicate"
        ),
    )


def feature_pyramid(tokens, height, width, stages):
    batch_size, _, channels = tokens.shape
    base = tokens.transpose(1, 2).reshape(
        batch_size, channels, height // 16, width // 16
    )
    aspect_ratio = width / height
    span_horizontal = aspect_ratio / (1 + aspect_ratio**2) ** 0.5
    span_vertical = 1 / (1 + aspect_ratio**2) ** 0.5
    features = []
    for level in range(stages):
        level_height, level_width = height // 16 * 2**level, width // 16 * 2**level
        horizontal = torch.linspace(
            -span_horizontal * (level_width - 1) / level_width,
            span_horizontal * (level_width - 1) / level_width,
            level_width,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        vertical = torch.linspace(
            -span_vertical * (level_height - 1) / level_height,
            span_vertical * (level_height - 1) / level_height,
            level_height,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        vertical, horizontal = torch.meshgrid(vertical, horizontal, indexing="ij")
        coordinates = (
            torch.stack((horizontal, vertical))
            .unsqueeze(0)
            .expand(batch_size, -1, -1, -1)
        )
        features.append(
            torch.cat((base, coordinates), dim=1).contiguous()
            if level == 0
            else coordinates.contiguous()
        )
    return features


class ConvPyramid(nn.Module):
    def __init__(self, dimensions, output_channels):
        super().__init__()
        self.input_blocks = nn.ModuleList(
            nn.Conv2d(1026 if index == 0 else 2, channels, 1)
            for index, channels in enumerate(dimensions)
        )
        self.res_blocks = nn.ModuleList(
            nn.Sequential(ConvResidual(channels)) for channels in dimensions
        )
        self.resamplers = nn.ModuleList(
            upsample_block(first, second)
            for first, second in zip(dimensions[:-1], dimensions[1:])
        )
        self.output = nn.Conv2d(dimensions[-1], output_channels, 1)

    def forward(self, pyramid):
        with torch.autocast(device_type=pyramid[0].device.type, enabled=False):
            features = self.res_blocks[0](self.input_blocks[0](pyramid[0].float()))
            prefix = features
            for index in range(1, len(self.res_blocks)):
                features = self.resamplers[index - 1](features) + self.input_blocks[
                    index
                ](pyramid[index].float())
                features = self.res_blocks[index](features)
            return self.output(features), prefix
