import torch
from torch import nn
from torch.nn import functional as F


def partition(features, window):
    batch_size, height, width, channels = features.shape
    pad_height, pad_width = -height % window, -width % window
    features = F.pad(features, (0, 0, 0, pad_width, 0, pad_height))
    padded_height, padded_width = height + pad_height, width + pad_width
    features = features.reshape(
        batch_size,
        padded_height // window,
        window,
        padded_width // window,
        window,
        channels,
    )
    return features.permute(0, 1, 3, 2, 4, 5).reshape(-1, window, window, channels), (
        padded_height,
        padded_width,
    )


def unpartition(features, window, padded_size, original_size):
    padded_height, padded_width = padded_size
    height, width = original_size
    batch_size = features.shape[0] // (padded_height * padded_width // window**2)
    features = features.reshape(
        batch_size, padded_height // window, padded_width // window, window, window, -1
    )
    features = features.permute(0, 1, 3, 2, 4, 5).reshape(
        batch_size, padded_height, padded_width, -1
    )
    return features[:, :height, :width]


def pool(features):
    return F.max_pool2d(features.permute(0, 3, 1, 2), 2, 2).permute(0, 2, 3, 1)


class HieraMLP(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.ModuleList(
            (nn.Linear(channels, channels * 4), nn.Linear(channels * 4, channels))
        )

    def forward(self, features):
        return self.layers[1](F.gelu(self.layers[0](features)))


class HieraAttention(nn.Module):
    def __init__(self, dimension, output_dimension, heads, downsample):
        super().__init__()
        self.heads = heads
        self.downsample = downsample
        self.qkv = nn.Linear(dimension, 3 * output_dimension)
        self.proj = nn.Linear(output_dimension, output_dimension)

    def forward(self, features):
        batch_size, height, width, _ = features.shape
        query, key, value = (
            self.qkv(features)
            .reshape(batch_size, height * width, 3, self.heads, -1)
            .unbind(2)
        )
        if self.downsample:
            query = pool(query.reshape(batch_size, height, width, -1))
            height, width = query.shape[1:3]
            query = query.reshape(batch_size, height * width, self.heads, -1)
        features = F.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        )
        return self.proj(
            features.transpose(1, 2).reshape(batch_size, height, width, -1)
        )


class HieraBlock(nn.Module):
    def __init__(self, dimension, output_dimension, heads, window, downsample):
        super().__init__()
        self.window = window
        self.downsample = downsample
        self.norm1 = nn.LayerNorm(dimension, eps=1e-6)
        self.norm2 = nn.LayerNorm(output_dimension, eps=1e-6)
        self.attn = HieraAttention(dimension, output_dimension, heads, downsample)
        self.mlp = HieraMLP(output_dimension)
        self.proj = (
            nn.Linear(dimension, output_dimension)
            if dimension != output_dimension
            else None
        )

    def forward(self, features):
        shortcut = features
        features = self.norm1(features)
        if self.proj is not None:
            shortcut = pool(self.proj(features))
        height, width = features.shape[1:3]
        if self.window:
            features, padded_size = partition(features, self.window)
        features = self.attn(features)
        window = self.window
        if self.downsample:
            window //= 2
            height, width = shortcut.shape[1:3]
            padded_size = (height + (-height % window), width + (-width % window))
        if self.window:
            features = unpartition(features, window, padded_size, (height, width))
        features = shortcut + features
        return features + self.mlp(self.norm2(features))


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, 96, 7, stride=4, padding=3)
        self.pos_embed = nn.Parameter(torch.empty(1, 96, 7, 7))
        self.pos_embed_window = nn.Parameter(torch.empty(1, 96, 8, 8))
        self.blocks = nn.ModuleList()
        dimension, heads, stage = 96, 1, 0
        for index in range(16):
            window = (8, 4, 14, 7)[stage]
            if index in (7, 10, 13):
                window = 0
            downsample = index in (1, 3, 14)
            output_dimension = dimension * 2 if downsample else dimension
            if downsample:
                heads *= 2
                stage += 1
            self.blocks.append(
                HieraBlock(dimension, output_dimension, heads, window, downsample)
            )
            dimension = output_dimension
        self.fpn = nn.ModuleList(
            nn.Conv2d(channels, 256, 1) for channels in (768, 384, 192, 96)
        )

    def forward(self, images):
        features = self.patch_embed(images)
        height, width = features.shape[-2:]
        position = F.interpolate(self.pos_embed, size=(height, width), mode="bicubic")
        position = position + self.pos_embed_window.tile(1, 1, height // 8, width // 8)
        features = (features + position).permute(0, 2, 3, 1)
        pyramid = []
        for index, block in enumerate(self.blocks):
            features = block(features)
            if index in (0, 2, 13, 15):
                pyramid.append(features.permute(0, 3, 1, 2))
        output = [None] * 4
        previous = None
        for index in range(3, -1, -1):
            lateral = self.fpn[3 - index](pyramid[index])
            previous = (
                lateral
                + F.interpolate(previous.float(), scale_factor=2, mode="nearest")
                if index in (2, 3) and previous is not None
                else lateral
            )
            output[index] = previous
        return output[:3]


class ChannelLayerNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, features):
        return F.layer_norm(
            features.permute(0, 2, 3, 1),
            (features.shape[1],),
            self.weight,
            self.bias,
            1e-6,
        ).permute(0, 3, 1, 2)


class MotionMask(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ImageEncoder()
        self.motion_projection = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, 256))
        self.fusion = nn.Conv2d(512, 256, 1)
        self.residual = nn.Sequential(
            nn.GroupNorm(32, 256),
            nn.GELU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.GroupNorm(32, 256),
            nn.GELU(),
            nn.Conv2d(256, 256, 3, padding=1),
        )
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 2, stride=2),
            ChannelLayerNorm(64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.GELU(),
        )
        self.conv_s0 = nn.Conv2d(256, 32, 1)
        self.conv_s1 = nn.Conv2d(256, 64, 1)
        self.output = nn.Conv2d(32, 1, 1)

    def forward(self, tokens, images):
        height, width = images.shape[-2:]
        outputs = []
        for frame_tokens, frame_image in zip(tokens.split(1), images.split(1)):
            mean = frame_image.new_tensor((0.485, 0.456, 0.406)).reshape(1, 3, 1, 1)
            deviation = frame_image.new_tensor((0.229, 0.224, 0.225)).reshape(
                1, 3, 1, 1
            )
            rgb = (frame_image.float() * deviation + mean).clamp(0, 1)
            resized = F.interpolate(
                rgb,
                size=(1024, 1024),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            fine, middle, coarse = self.encoder((resized - mean) / deviation)
            motion = (
                self.motion_projection(frame_tokens)
                .transpose(1, 2)
                .reshape(1, 256, height // 16, width // 16)
            )
            motion = F.interpolate(
                motion, size=coarse.shape[-2:], mode="bilinear", align_corners=False
            )
            features = self.fusion(torch.cat((motion, coarse.to(motion.dtype)), dim=1))
            features = features + self.residual(features)
            features = self.output_upscaling[2](
                self.output_upscaling[1](
                    self.output_upscaling[0](features) + self.conv_s1(middle)
                )
            )
            features = self.output_upscaling[4](
                self.output_upscaling[3](features) + self.conv_s0(fine)
            )
            outputs.append(
                F.interpolate(
                    self.output(features).float(),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )[:, 0]
            )
        return torch.cat(outputs)
