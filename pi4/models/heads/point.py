from torch import nn

from ..layers import ConvPyramid, ConvResidual, feature_pyramid, upsample_block


class ConfidenceHead(nn.Module):
    def __init__(self):
        super().__init__()
        dimensions = (256, 128, 64, 32)
        self.input_blocks = nn.ModuleList(
            nn.Conv2d(2, channels, 1) for channels in dimensions
        )
        self.res_blocks = nn.ModuleList(
            nn.Sequential(ConvResidual(channels)) for channels in dimensions
        )
        self.resamplers = nn.ModuleList(
            upsample_block(first, second)
            for first, second in zip((512, *dimensions[:-1]), dimensions)
        )
        self.output = nn.Conv2d(32, 1, 1)

    def forward(self, prefix, pyramid):
        features = prefix
        for index in range(4):
            features = self.resamplers[index](features) + self.input_blocks[index](
                pyramid[index + 1]
            )
            features = self.res_blocks[index](features)
        return self.output(features)


class PointHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.points = ConvPyramid((512, 256, 128, 64, 32), 3)
        self.confidence = ConfidenceHead()

    def forward(self, tokens, height, width):
        pyramid = feature_pyramid(tokens.float(), height, width, 5)
        coordinates, prefix = self.points(pyramid)
        logits = self.confidence(prefix, pyramid)[:, 0]
        coordinates = coordinates.permute(0, 2, 3, 1)
        rays, log_depth = coordinates[..., :2], coordinates[..., 2]
        return rays, log_depth, logits, pyramid[0]
