import torch

from .geometry import depth_normal_edge


def geometry_confidence(points, confidence):
    """Copy geometry confidence, zeroing invalid pixels and depth-normal edges."""
    valid = (
        torch.isfinite(points).all(dim=-1)
        & (points[..., 2] > 0)
        & torch.isfinite(confidence)
        & (confidence >= 0)
    )
    edges = depth_normal_edge(points, valid)
    return confidence.masked_fill(~valid | edges, 0)
