# Example Videos

These are RGB input videos for Pi4 inference, not precomputed model outputs.
Run commands from the repository root; weights are downloaded from `yyfz233/Pi4`
automatically. For a manually downloaded checkpoint, add
`--weights /path/to/model.safetensors`.

| Video | Stored / inference size | Source frames | Sampling | Sampled frames | Coverage |
| --- | --- | --- | --- | --- | --- |
| `figure_skating.mp4` | 512×288 | 750 at 12.5 fps | Stride 1, optional cap 150 | 150 | First approximately 12 seconds |
| `kart.mp4` | 512×288 | 300 at 15000/1001 fps | Stride 2 | 150 | Full approximately 20-second clip |
| `cycling.mp4` | 288×512 | 356 at 30 fps | Stride 2 | 178 | Full approximately 11.87-second clip |

All three MP4s use H.264 (`libx264`, CRF 20, `slow` preset), YUV420p, and
video-only streams with fast-start metadata for browser playback.
They are compressed at the same patch-aligned resolution used
by the default 512-pixel inference loader. Every source frame and its timing are
retained: `--stride` samples frames during loading, not in the stored video.
`--max-frames` is optional and caps frames **after** sampling. Without it, the
CLI processes the complete sampled sequence. Each command runs one joint
forward pass, without temporal windows.

## Figure Skating

```bash
python example.py examples/figure_skating.mp4 \
  --output outputs/quickstart \
  --query-frame 0 --stride 1 --max-frames 150 --viewer
```

This uses source frames 0–149: the first approximately 12 seconds of the
60-second clip. The full video is long, so the quickstart uses an optional
150-frame cap to reduce runtime and GPU memory use. To reproduce the full
Figure Skating demo from the paper and project page, keep stride 1 and omit
`--max-frames` to process all 750 frames without subsampling:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
  python example.py examples/figure_skating.mp4 \
  --output outputs/figure_skating_full \
  --query-frame 0 --stride 1 --viewer
```

The full run requires substantially more GPU memory. See the
[long-sequence measurements](../docs/inference.md#reproduce-the-project-page-demo).

## Kart — Indoor Karting

```bash
python example.py examples/kart.mp4 \
  --output outputs/kart \
  --query-frame 0 --stride 2 --viewer
```

This samples source frames 0, 2, ..., 298, covering the complete 300-frame clip
at approximately 7.49 sampled frames per second. The stored video is 512×288,
matching the previous inference resize from the original 384×216 input; it does
not contain additional source detail.

## Cycling

```bash
python example.py examples/cycling.mp4 \
  --output outputs/cycling \
  --query-frame 0 --stride 2 --viewer
```

This samples source frames 0, 2, ..., 354, covering the complete 356-frame clip
at 15 sampled frames per second. Its portrait input is stored at 288×512 rather
than the original 720×1280. The 178-frame run is larger than the 150-frame
Figure Skating quickstart; GPU memory use depends on the complete sampled sequence.

## Open Results

```bash
python viewer/serve.py outputs/quickstart --port 8080
python viewer/serve.py outputs/kart --port 8081
python viewer/serve.py outputs/cycling --port 8082
```

Run the appropriate server for the result you want to inspect. `--viewer` is
optional for inference; omit it for regular PLY/camera/track export.

Figure Skating and Kart derive from the paper website's RGB input assets;
Cycling is the supplied cycling clip. No website point clouds, tracks, or rendered result videos
are included. Compression is lossy, so predictions need not be numerically
identical to runs using the previous video encodings.
