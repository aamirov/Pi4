"""Visualization-only sampling, masking, and scene export."""

import colorsys
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .io import (
    _encode_motion_masks,
    _numpy,
    _ply_header,
    _track_arrays,
    _write_bundle,
    _write_ply,
)
from .postprocess import geometry_confidence


MOTION_MASK_THRESHOLD = 0.5
BACKGROUND_MASK_DILATION_PIXELS = 5
TRAJECTORY_MASK_EROSION_PIXELS = 2
POINT_CONFIDENCE_THRESHOLD = 0.05
TRACK_CONFIDENCE_THRESHOLD = 0.5
TRACK_VISIBILITY_THRESHOLD = 0.5
MAX_TRACK_QUERIES = 1024
PREVIEW_MAX_BYTES = 10_000_000
VIEWER_MAX_BYTES = 40_000_000
VIEWER_CHUNK_FRAMES = 8


def _save_preview(output, frames, prediction, cameras, scale):
    import torch

    masks, finite_counts, eligible_counts = [], [], []
    for frame in range(len(frames)):
        points = prediction["local_points"][0, frame].detach().float().cpu()
        confidence = geometry_confidence(
            points, prediction["confidence"][0, frame].detach().float().cpu()
        )
        finite = torch.isfinite(points).all(dim=-1)
        keep = finite & (confidence >= POINT_CONFIDENCE_THRESHOLD)
        masks.append(keep.numpy().reshape(-1))
        finite_counts.append(int(finite.sum()))
        eligible_counts.append(int(keep.sum()))

    total = sum(eligible_counts)
    count = min(total, (PREVIEW_MAX_BYTES - len(_ply_header(total))) // 15)
    indices = np.linspace(0, total - 1, count, dtype=np.int64)
    vertices = np.empty(count, dtype=[("position", "<f4", (3,)), ("color", "u1", (3,))])
    offset, cursor = 0, 0
    for frame, (mask, eligible) in enumerate(zip(masks, eligible_counts)):
        selected = indices[(indices >= offset) & (indices < offset + eligible)] - offset
        if len(selected):
            pixels = np.flatnonzero(mask)[selected]
            points = _numpy(prediction["local_points"][0, frame]).reshape(-1, 3)[pixels]
            world = (points * scale) @ cameras[frame, :3, :3].T + cameras[frame, :3, 3]
            selected_vertices = slice(cursor, cursor + len(pixels))
            vertices["position"][selected_vertices] = world
            vertices["color"][selected_vertices] = frames[frame].reshape(-1, 3)[pixels]
            cursor += len(pixels)
        offset += eligible
    description = _write_ply(output / "pc_preview.ply", vertices)
    if description["bytes"] > PREVIEW_MAX_BYTES or cursor != count:
        raise RuntimeError("Preview size or sampling invariant failed")
    return dict(
        **description,
        confidence_threshold=POINT_CONFIDENCE_THRESHOLD,
        sampling="linspace_after_filtering",
        max_bytes=PREVIEW_MAX_BYTES,
        depth_normal_edge=dict(
            relative_depth_threshold=0.03, normal_threshold_degrees=5, kernel_size=3
        ),
        finite_points_per_frame=finite_counts,
        eligible_points_per_frame=eligible_counts,
    )


def _native_motion_mask(probability):
    return np.isfinite(probability) & (probability >= MOTION_MASK_THRESHOLD)


def _background_motion_mask(native_mask):
    radius = BACKGROUND_MASK_DILATION_PIXELS
    kernel_size = 2 * radius + 1
    horizontal = np.lib.stride_tricks.sliding_window_view(
        np.pad(native_mask, radius), kernel_size, axis=1
    ).any(axis=-1)
    return np.lib.stride_tricks.sliding_window_view(
        horizontal, kernel_size, axis=0
    ).any(axis=-1)


def _trajectory_motion_mask(probability):
    radius = TRAJECTORY_MASK_EROSION_PIXELS
    kernel_size = 2 * radius + 1
    padded = np.pad(_native_motion_mask(probability), radius, constant_values=False)
    return np.lib.stride_tricks.sliding_window_view(
        padded, (kernel_size, kernel_size)
    ).all(axis=(-2, -1))


def _track_data(frames, prediction, *, motion_only):
    frame_count, height, width = frames.shape[:3]
    query_frame = int(prediction["query_frame"][0])
    query_points = _numpy(prediction["tracks_3d"][0, query_frame]).reshape(-1, 3)
    eligible = np.isfinite(query_points).all(axis=-1)
    motion_only = motion_only and "dynamic_mask" in prediction
    if motion_only:
        eligible &= _trajectory_motion_mask(
            _numpy(prediction["dynamic_mask"][0, query_frame])
        ).reshape(-1)
    eligible = np.flatnonzero(eligible)
    indices = eligible[
        np.linspace(
            0, len(eligible) - 1, min(MAX_TRACK_QUERIES, len(eligible)), dtype=np.int64
        )
    ]
    arrays = _track_arrays(frames, prediction, indices)
    arrays["valid"] = (
        np.isfinite(arrays["world_xyz"]).all(axis=-1)
        & np.isfinite(arrays["pixel_xy"]).all(axis=-1)
        & np.isfinite(arrays["visibility"])
        & (arrays["visibility"] >= TRACK_VISIBILITY_THRESHOLD)
    ).astype("u1")
    if motion_only:
        for frame in range(frame_count):
            pixels = arrays["pixel_xy"][frame]
            inside = (
                np.isfinite(pixels).all(axis=-1)
                & (pixels[:, 0] >= 0)
                & (pixels[:, 0] <= width - 1)
                & (pixels[:, 1] >= 0)
                & (pixels[:, 1] <= height - 1)
            )
            arrays["valid"][frame] &= inside
            selected = np.flatnonzero(arrays["valid"][frame])
            coordinates = np.floor(pixels[selected] + 0.5).astype(np.int64)
            mask = _trajectory_motion_mask(
                _numpy(prediction["dynamic_mask"][0, frame])
            )
            arrays["valid"][frame, selected] &= mask[
                coordinates[:, 1], coordinates[:, 0]
            ]
    return arrays, dict(
        query_frame=query_frame,
        query_count=len(indices),
        visibility_threshold=TRACK_VISIBILITY_THRESHOLD,
        motion_mask_filter=(
            dict(
                threshold=MOTION_MASK_THRESHOLD,
                erosion_kernel_size=2 * TRAJECTORY_MASK_EROSION_PIXELS + 1,
                query_and_target=True,
                pixel_sampling="nearest",
            )
            if motion_only
            else None
        ),
    )


def _query_colors(query_uv):
    if not len(query_uv):
        return np.empty((0, 3), dtype="u1")
    span = np.ptp(query_uv, axis=0)
    normalized = np.full(query_uv.shape, 0.5, dtype=np.float64)
    np.divide(query_uv - query_uv.min(axis=0), span, out=normalized, where=span > 0)
    colors = np.array(
        [
            colorsys.hls_to_rgb(
                0.72 * (0.7 * horizontal + 0.3 * vertical),
                0.53 + 0.08 * vertical,
                0.82,
            )
            for horizontal, vertical in normalized
        ]
    )
    colors = np.where(
        colors <= 0.0031308, 12.92 * colors, 1.055 * colors ** (1 / 2.4) - 0.055
    )
    return np.rint(np.clip(colors, 0, 1) * 255).astype("u1")


def _save_track_preview(output, arrays, description):
    confidence = arrays["confidence"]
    valid = (
        arrays["valid"].astype(bool)
        & np.isfinite(confidence)
        & (confidence >= TRACK_CONFIDENCE_THRESHOLD)
    )
    vertices = np.empty(
        int(valid.sum()), dtype=[("position", "<f4", (3,)), ("color", "u1", (3,))]
    )
    vertices["position"] = arrays["world_xyz"][valid]
    vertices["color"] = np.broadcast_to(
        _query_colors(arrays["query_uv"]), arrays["world_xyz"].shape
    )[valid]
    return dict(
        **_write_ply(output / "tracks.ply", vertices),
        **description,
        confidence_threshold=TRACK_CONFIDENCE_THRESHOLD,
        colors="query_xy_hsl",
        order="frame_then_query",
    )


def _save_tracks(output, frames, prediction, *, viewer=False):
    if "tracks_3d" not in prediction:
        return None
    arrays, description = _track_data(frames, prediction, motion_only=viewer)
    description.update(_write_bundle(output, "tracks.bin", arrays))
    if not viewer:
        motion_arrays, motion_description = _track_data(
            frames, prediction, motion_only=True
        )
        description["preview"] = _save_track_preview(
            output, motion_arrays, motion_description
        )
    return description


def _viewer_motion_flags(probability):
    native_mask = _native_motion_mask(probability)
    background_mask = _background_motion_mask(native_mask)
    return _encode_motion_masks(native_mask, background_mask)


def _save_viewer(output, frames, prediction, records, cameras, scale):
    max_bytes = VIEWER_MAX_BYTES
    bytes_per_point = 20
    chunk_frames = VIEWER_CHUNK_FRAMES
    frame_count = len(frames)
    point_limit = min(
        frames.shape[1] * frames.shape[2],
        max_bytes // (frame_count * bytes_per_point),
    )
    padding = 4 if (frame_count % chunk_frames) * point_limit % 4 else 0
    if frame_count * point_limit * bytes_per_point + padding > max_bytes:
        point_limit -= 1
    (output / "frames").mkdir()
    (output / "points").mkdir()
    chunks = []
    has_mask = "dynamic_mask" in prediction
    for start in range(0, frame_count, chunk_frames):
        positions, colors, motion, confidences = [], [], [], []
        offset = 0
        for frame in range(start, min(start + chunk_frames, frame_count)):
            points = _numpy(prediction["local_points"][0, frame]).reshape(-1, 3)
            finite = np.flatnonzero(np.isfinite(points).all(axis=-1))
            indices = finite[
                np.linspace(0, len(finite) - 1, min(point_limit, len(finite)), dtype=np.int64)
            ]
            world = (points[indices] * scale) @ cameras[frame, :3, :3].T + cameras[
                frame, :3, 3
            ]
            positions.append(world.astype("<f4"))
            colors.append(frames[frame].reshape(-1, 3)[indices])
            confidence = geometry_confidence(
                prediction["local_points"][0, frame].detach().float().cpu(),
                prediction["confidence"][0, frame].detach().float().cpu(),
            )
            confidences.append(confidence.numpy().reshape(-1)[indices].astype("<f4"))
            flags = np.zeros(len(indices), dtype="u1")
            if has_mask:
                flags = _viewer_motion_flags(
                    _numpy(prediction["dynamic_mask"][0, frame])
                ).reshape(-1)[indices]
            motion.append(flags)
            image = f"frames/{frame:06d}.jpg"
            Image.fromarray(frames[frame]).save(output / image, quality=85)
            records[frame].update(
                chunk=len(chunks),
                start=offset,
                count=len(indices),
                finite_points=len(finite),
                image=image,
            )
            offset += len(indices)
        chunks.append(
            _write_bundle(
                output,
                f"points/{start:06d}.bin",
                dict(
                    position=np.concatenate(positions),
                    color=np.concatenate(colors),
                    motion=np.concatenate(motion),
                    confidence=np.concatenate(confidences),
                ),
            )
        )
    total_bytes = sum(chunk["bytes"] for chunk in chunks)
    if total_bytes > max_bytes:
        raise RuntimeError("Viewer point-chunk budget exceeded")
    return dict(
        chunks=chunks,
        max_points_per_frame=point_limit,
        max_bytes=max_bytes,
        bytes=total_bytes,
        bytes_per_point=bytes_per_point,
        points=sum(record["count"] for record in records),
        sampling="linspace",
        chunk_frames=chunk_frames,
        pointcloud_mode="confidence_filterable_preview",
        confidence=dict(
            source="geometry_confidence",
            default_threshold=POINT_CONFIDENCE_THRESHOLD,
            depth_normal_edge=dict(
                relative_depth_threshold=0.03,
                normal_threshold_degrees=5,
                kernel_size=3,
                confidence_value=0,
            ),
        ),
        has_motion_mask=has_mask,
        motion_mask_threshold=MOTION_MASK_THRESHOLD if has_mask else None,
        motion_mask_dilation_pixels=BACKGROUND_MASK_DILATION_PIXELS if has_mask else None,
        motion_mask_encoding="native_bit_0_dilated_bit_1" if has_mask else None,
    )


def save_visualization(output, frames, prediction, metadata, *, viewer=False):
    """Save sampled previews and optional viewer data without changing predictions."""
    frame_count, height, width, channels = frames.shape
    if (
        frames.dtype != np.uint8
        or channels != 3
        or min(frame_count, height, width) < 1
        or prediction["local_points"].shape != (1, frame_count, height, width, 3)
        or prediction["confidence"].shape != (1, frame_count, height, width)
    ):
        raise ValueError(
            "Expected RGB uint8 [T,H,W,3] and matching single-sequence predictions"
        )
    cameras = _numpy(prediction["camera_poses"][0]).copy()
    scale = float(prediction["metric_scale"][0])
    if (
        cameras.shape != (frame_count, 4, 4)
        or not np.isfinite(cameras).all()
        or not np.isfinite(scale)
        or scale <= 0
    ):
        raise ValueError(
            "Expected finite camera-to-world poses and a positive metric scale"
        )
    cameras[:, :3, 3] *= scale
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    records = [dict(camera_to_world=camera.tolist()) for camera in cameras]
    scene = dict(
        format="pi4_scene_v1",
        coordinates="metric_world",
        camera_convention="camera_to_world",
        image_size=[width, height],
        metric_scale=scale,
        source=metadata,
        frames=records,
        preview=_save_preview(output, frames, prediction, cameras, scale),
        tracks=_save_tracks(output, frames, prediction, viewer=viewer),
        viewer=None,
    )
    if viewer:
        scene["viewer"] = _save_viewer(
            output, frames, prediction, records, cameras, scale
        )
    (output / "scene.json").write_text(
        json.dumps(scene, indent=2, allow_nan=False) + "\n"
    )
    return scene
