"""Image/video loading and serialization primitives, without display filtering."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def load_images(source, *, stride=1, max_frames=None, max_size=512):
    source = Path(source)
    if stride < 1 or max_size < 16 or (max_frames is not None and max_frames < 1):
        raise ValueError("stride, max_frames, and max_size must be positive")
    frames, indices, source_metadata = [], [], {}
    if source.is_dir():
        files = sorted(
            path
            for path in source.iterdir()
            if path.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp")
        )
        for index in range(0, len(files), stride):
            with Image.open(files[index]) as image:
                frames.append(ImageOps.exif_transpose(image).convert("RGB"))
            indices.append(index)
            if max_frames is not None and len(frames) == max_frames:
                break
        source_metadata["source_files"] = [files[index].name for index in indices]
    elif source.is_file() and source.suffix.lower() in (
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp",
    ):
        with Image.open(source) as image:
            frames.append(ImageOps.exif_transpose(image).convert("RGB"))
        indices.append(0)
    elif source.is_file():
        import cv2

        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {source}")
        frame_rate = capture.get(cv2.CAP_PROP_FPS)
        index = 0
        try:
            while True:
                available, frame = capture.read()
                if not available:
                    break
                if index % stride == 0:
                    frames.append(
                        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    )
                    indices.append(index)
                    if max_frames is not None and len(frames) == max_frames:
                        break
                index += 1
        finally:
            capture.release()
        if np.isfinite(frame_rate) and frame_rate > 0:
            source_metadata["source_fps"] = frame_rate
            source_metadata["source_timestamps_s"] = [
                index / frame_rate for index in indices
            ]
    else:
        raise FileNotFoundError(source)
    if not frames:
        raise ValueError(f"No images found in {source}")
    width, height = frames[0].size
    scale = max_size / max(height, width)
    target = (
        max(16, round(width * scale / 16) * 16),
        max(16, round(height * scale / 16) * 16),
    )
    frames = [
        np.asarray(frame.resize(target, Image.Resampling.LANCZOS)) for frame in frames
    ]
    return np.stack(frames), dict(
        source_metadata,
        source=str(source.resolve()),
        source_frame_indices=indices,
        image_size=list(target),
        resize="shared first-image aspect ratio; patch-aligned resize",
    )


def _numpy(tensor):
    return tensor.detach().float().cpu().numpy()


def _write_bundle(output, filename, arrays):
    description = dict(file=filename, bytes=0, arrays={})
    with (output / filename).open("wb") as stream:
        for name, values in arrays.items():
            values = np.ascontiguousarray(values)
            stream.write(b"\0" * (-stream.tell() % 4))
            description["arrays"][name] = dict(
                dtype=values.dtype.str,
                shape=list(values.shape),
                offset=stream.tell(),
                bytes=values.nbytes,
            )
            values.tofile(stream)
        description["bytes"] = stream.tell()
    return description


def _ply_header(count):
    return (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {count}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    ).encode("ascii")


def _write_ply(filename, vertices):
    with filename.open("wb") as stream:
        stream.write(_ply_header(len(vertices)))
        vertices.tofile(stream)
    return dict(file=filename.name, points=len(vertices), bytes=filename.stat().st_size)


def _track_arrays(frames, prediction, indices):
    frame_count, height, width = frames.shape[:3]
    query_frame = int(prediction["query_frame"][0])
    arrays = dict(
        query_uv=np.stack((indices % width, indices // width), axis=-1).astype("<f4"),
        query_rgb=frames[query_frame].reshape(-1, 3)[indices],
    )
    for source, target, channels in (
        ("tracks_3d", "world_xyz", (3,)),
        ("tracks_2d", "pixel_xy", (2,)),
        ("visibility", "visibility", ()),
        ("tracking_confidence", "confidence", ()),
    ):
        arrays[target] = np.stack(
            [
                _numpy(prediction[source][0, frame]).reshape(height * width, *channels)[
                    indices
                ]
                for frame in range(frame_count)
            ]
        ).astype("<f4")
    return arrays


def _encode_motion_masks(native_mask, background_mask):
    """Pack native motion in bit 0 and background exclusion in bit 1."""
    return native_mask.astype("u1") | (background_mask.astype("u1") << 1)


def save_results(output, frames, prediction, metadata, *, viewer=False):
    """Compatibility entry point for pi4.visualization.save_visualization."""
    from .visualization import save_visualization

    return save_visualization(output, frames, prediction, metadata, viewer=viewer)
