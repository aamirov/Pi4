import math

import torch
from torch.nn import functional as F


def depth_normal_edge(points, mask):
    """3x3 depth jumps >3% AND local normal discontinuities >5 degrees."""
    height, width = points.shape[:2]
    valid = mask & torch.isfinite(points).all(dim=-1) & (points[..., 2] > 0)
    depth = points[..., 2][None, None]
    maximum = F.max_pool2d(torch.where(valid, depth, -torch.inf), 3, 1, 1)
    minimum = -F.max_pool2d(torch.where(valid, -depth, -torch.inf), 3, 1, 1)
    depth_edges = ((maximum - minimum) / depth).nan_to_num()[0, 0] > 0.03

    safe_points = torch.where(valid[..., None], points, 0)
    padded = F.pad(safe_points.permute(2, 0, 1), (1, 1, 1, 1))
    padded = padded.permute(1, 2, 0)
    valid_pad = F.pad(valid, (1, 1, 1, 1))
    neighbors = [
        padded[:-2, 1:-1], padded[1:-1, :-2],
        padded[2:, 1:-1], padded[1:-1, 2:],
    ]
    neighbor_valid = [
        valid_pad[:-2, 1:-1], valid_pad[1:-1, :-2],
        valid_pad[2:, 1:-1], valid_pad[1:-1, 2:],
    ]
    normals = torch.zeros_like(points)
    normal_valid = torch.zeros_like(valid)
    for index in range(4):
        following = (index + 1) % 4
        keep = valid & neighbor_valid[index] & neighbor_valid[following]
        cross = torch.linalg.cross(
            neighbors[index] - points, neighbors[following] - points
        )
        normals += torch.where(
            keep[..., None], F.normalize(cross, dim=-1, eps=1e-12), 0
        )
        normal_valid |= keep
    normals = F.normalize(normals, dim=-1, eps=1e-12).nan_to_num()
    padded = F.pad(
        normals.permute(2, 0, 1)[None], (1, 1, 1, 1), mode="replicate"
    )[0].permute(1, 2, 0)
    valid_pad = F.pad(
        normal_valid[None, None].float(), (1, 1, 1, 1), mode="replicate"
    )[0, 0].bool()
    normal_edges = torch.zeros_like(valid)
    for vertical in range(3):
        for horizontal in range(3):
            neighbor = padded[
                vertical:vertical + height, horizontal:horizontal + width
            ]
            keep = valid_pad[
                vertical:vertical + height, horizontal:horizontal + width
            ]
            dot = (normals * neighbor).sum(dim=-1).clamp(-1, 1)
            normal_edges |= keep & (dot < math.cos(math.radians(5)))
    normal_edges = F.max_pool2d(
        normal_edges[None, None].float(), 3, 1, 1
    )[0, 0].bool()
    return depth_edges & normal_edges & normal_valid & valid
