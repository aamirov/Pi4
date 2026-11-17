---
title: Pi4
emoji: 🌌
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12"
app_file: app.py
pinned: false
license: cc-by-nc-4.0
short_description: Geometry and motion from RGB images and videos
models:
  - yyfz233/Pi4
---

# π⁴: A Foundation Model for 4D Scene Geometry and Motion

**Physical Intelligence Team · Shanghai AI Laboratory**

Upload a video or a collection of images to reconstruct geometry and dense motion
without supplied camera calibration. The demo reuses Pi4's native interactive
viewer for point clouds, camera poses, and dynamic tracks.

## Using the Demo

The outer interface follows the project website's light design: a compact
project header with a **Project Page** link and a full-width white scene viewer.
The input card and example cards sit below the viewer, not beside it. Video and
image uploads share one input area. Example clips have real video covers and
one-click loading; selecting one does not start inference automatically. Advanced
sampling options and result downloads are collapsed to keep the workspace clear.
The page stays light even when the surrounding Hugging Face UI uses dark mode.

1. Choose **Video** or **Images**, then upload media or select a bundled example.
2. The hosted demo uses stride **1**, up to **150** sampled frames,
   and a fixed **512-pixel** maximum image side. Figure Skating uses frames 0–149,
   covering about 12 seconds. Unlike the CLI, the Space always caps sampled
   frames to limit hosted GPU use. The Kart example selects stride **2**, covering
   its full approximately 20-second clip with 150 frames. Change the stride to cover more
   of a video, or lower the frame limit to reduce GPU memory use. Images are
   ordered by filename.
3. Enable tracking for geometry and motion, or disable it for geometry only.
   A single image requires geometry-only mode. The tracking query index refers
   to the sampled frames, not the original video.
4. Click **Reconstruct & Track**, then explore the embedded viewer or open it in
   a new tab. Download the preview PLY or the complete scene ZIP.

The viewer follows the project website's light, purple-accented design, with a
synchronized RGB inset, filmstrip timeline, playback speed, camera frustums,
fullscreen mode, and collapsible geometry/trajectory filters. Click or drag the
filmstrip, or click the RGB inset, to smoothly return to the selected predicted
camera. Scrubbing pauses playback while dragging and resumes it only if it was
already playing. The render-budget slider uses
deterministic `linspace` sampling, defaults to up to **2,000,000** exported
points, and actually reduces the GPU draw count when lowered. The existing
40 MB geometry export budget limits the available points; small scenes contain
fewer than two million. Visibility/confidence filters can reduce the number
shown further. Display changes never alter inference or downloaded results.

The viewer prefetches 32 future frames with up to three concurrent bulk
downloads (including tracks), instead of waiting at each eight-frame boundary.
Tracks load in the background and their controls become available when ready;
they do not block the initial scene or playback. RGB previews are decoded ahead
of playback in a bounded 29-frame cache, and point chunks are applied in small
cooperative batches to keep the page responsive. These changes do not lower
the point budget or change inference. A connection slower than the scene's
required data rate can still need buffering.

All sampled frames are processed together in one forward pass. The demo does
not silently reduce resolution, split the input into temporal windows, or retry
after an out-of-memory error. Reduce the frame limit if necessary.
The displayed forward time excludes GPU allocation, model transfer, and export.

Extract the scene ZIP and use the source repository's viewer to open it locally:

```bash
python viewer/serve.py /path/to/extracted-scene --port 8080
```

## Runtime

Select **ZeroGPU** in this Space's hardware settings. The Space uses the Gradio
SDK; only the CUDA forward pass, model transfer, and copying predictions back
to CPU run inside `@spaces.GPU(duration=120)`. Frame decoding, CPU weight loading,
and scene export do not reserve a GPU. GPU calls are serialized and queued.
Access remains subject to Hugging Face's current quota and organization rules.

The model is downloaded at runtime and cached on CPU between requests. The
first run may take longer while weights are downloaded and loaded. The
dependencies follow the source repository's PyTorch 2.12.1 / Triton 3.7.1 pins
and pinned FlexGEMM revision. The Space uses
the default PyTorch wheel, not the repository's optional CUDA 12.6 requirements;
the selected ZeroGPU hardware must support its CUDA build and sparse kernels.
An actual ZeroGPU forward pass is required to validate the hosted runtime.

For private model weights, add `HF_TOKEN` as a **Secret** in the Space settings.
Use a fine-grained token with read access to `yyfz233/Pi4` (or the repository
selected by `PI4_MODEL_ID`). The runtime downloader reads this secret
automatically; do not commit the token or put it in a public Variable.
`preload_from_hub` is intentionally omitted because it does not support private
repositories. Model downloads happen before GPU allocation.

Optional Space variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `PI4_MODEL_ID` | `yyfz233/Pi4` | Model repository or local checkpoint |
| `PI4_MODEL_REVISION` | unset | Optional checkpoint revision |
| `PI4_OUTPUT_DIR` | `outputs/space` | Temporary inference results |

Outputs older than six hours are removed when the next request starts; Gradio
also periodically expires its upload/download cache. Files disappear when the
Space restarts. Media and results are not private storage: anyone with a result
URL may access it until it expires. Do not upload sensitive material.

## License

Original Pi4 code and model weights are licensed under **CC BY-NC 4.0**.
Third-party materials retain their own terms, including applicable
CoWTracker-derived noncommercial research restrictions. See `LICENSE`,
`LICENSE_MODEL`, `NOTICE`, and `licenses/` for the full terms.

Source: https://github.com/yyfz/Pi4

Model: https://huggingface.co/yyfz233/Pi4
