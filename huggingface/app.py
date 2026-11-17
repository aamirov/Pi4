import gc
from functools import lru_cache, partial
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time
import uuid

import spaces
import gradio as gr
import torch
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from starlette.middleware import Middleware
from starlette.middleware.gzip import GZipMiddleware


HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / "pi4").is_dir() else HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
VIEWER = ROOT / "viewer"
OUTPUTS = Path(os.environ.get("PI4_OUTPUT_DIR", ROOT / "outputs" / "space")).resolve()
RETENTION_SECONDS = 6 * 60 * 60
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
RUN_ID = re.compile(r"[0-9a-f]{32}")
LOGGER = logging.getLogger(__name__)
STYLE = (HERE / "space.css").read_text()
EXAMPLE_COVERS = json.loads((HERE / "example-covers.json").read_text())
EXAMPLES = (
    ("figure_skating", "Figure Skating", "Articulated motion in an outdoor scene", "figure_skating.mp4", "150 frames · ~12 s"),
    ("kart", "Kart · driving", "A moving camera on an indoor circuit", "kart.mp4", "150 frames · ~20 s"),
)
HEADER = """
<header class="site-header">
  <div class="brand-lockup">
    <svg class="paper-logo" viewBox="0 0 952 762" aria-label="Pi4" role="img">
      <defs><linearGradient id="space-logo-gradient" x1="0%" y1="100%" x2="100%" y2="0%">
        <stop offset="0%" stop-color="#2563eb"/><stop offset="52%" stop-color="#7c3aed"/>
        <stop offset="100%" stop-color="#c026d3"/>
      </linearGradient></defs>
      <g fill="url(#space-logo-gradient)">
        <path transform="translate(0 740) scale(1 -1)" d="M171 485C139 485 101 485 79 468C55 450 31 355 18 291L46 279C62 325 92 388 135 388H229C221 333 200 243 180 200C145 125 89 40 75 21L76 8L154 -4L167 4C221 131 258 254 283 388H411L418 380C410 315 404 248 404 183C404 96 399 1 507 1C566 1 615 48 654 85L644 105L634 107C620 94 598 81 582 81C511 81 509 174 509 258C509 294 513 344 518 388H596C626 388 661 461 685 511L669 523C642 493 589 485 543 485Z"/>
        <path transform="translate(660 320) scale(.45 -.45)" d="M606 154V210H497V627C497 657 492 662 462 662H447C428 662 420 662 407 647L40 210V154H360V56H236V0C285 4 375 4 427 4C449 4 581 2 606 0V56H497V154ZM373 524V210H109Z"/>
      </g>
    </svg>
    <div class="brand-lab">Physical Intelligence Team<span>Shanghai AI Laboratory</span></div>
  </div>
  <nav aria-label="Project links">
    <a href="https://yyfz.github.io/pi4" target="_blank" rel="noopener">Project Page <span>↗</span></a>
    <a href="https://github.com/yyfz/Pi4" target="_blank" rel="noopener">GitHub <span>↗</span></a>
    <a href="https://huggingface.co/yyfz233/Pi4" target="_blank" rel="noopener">Model weights <span>↗</span></a>
  </nav>
</header>
<section class="page-intro">
  <div class="intro-copy">
    <p class="eyebrow">THE INTERACTIVE DEMO</p>
    <h1>π⁴: A Foundation Model<br><span>for 4D Scene Geometry and Motion</span></h1>
    <p class="intro-description">Reconstruct a scene and follow its motion from RGB video or images.<br>No camera calibration required.</p>
  </div>
  <div class="intro-note"><span class="live-dot"></span> POWERED BY ZEROGPU<span class="intro-note-detail">One model. Space and time, together.</span></div>
</section>
"""
EMPTY_VIEWER = """
<div class="empty-scene">
  <div class="empty-scene-icon" aria-hidden="true">
    <svg viewBox="0 0 96 96" fill="none">
      <path d="m48 16 29 17v32L48 82 19 65V33Z" stroke="#d4c4ec" stroke-width="1.2"/>
      <path d="m19 33 29 17 29-17M48 50v32M48 16v34" stroke="#d4c4ec" stroke-width="1.2"/>
      <path d="m35 39 13-8 13 8v16l-13 8-13-8Z" fill="#f0eafa" stroke="#9b79c6" stroke-width="1.3"/>
      <path d="m35 39 13 8 13-8M48 47v16" stroke="#9b79c6" stroke-width="1.3"/>
      <circle cx="19" cy="33" r="2.5" fill="#bda2df"/><circle cx="77" cy="65" r="2.5" fill="#bda2df"/>
      <circle cx="48" cy="16" r="2.5" fill="#bda2df"/>
    </svg>
  </div>
  <p class="eyebrow">A NEW PERSPECTIVE AWAITS</p>
  <h2>Your scene, in another dimension.</h2>
  <p>Upload a clip or try an example.<br>Explore geometry, cameras, and motion right here.</p>
  <div class="empty-scene-steps"><span><b>01</b> Add media</span><i></i><span><b>02</b> Reconstruct</span><i></i><span><b>03</b> Explore</span></div>
</div>
"""


def create_theme():
    palette = {
        "body_background_fill": "#f7f7f9",
        "background_fill_primary": "#ffffff",
        "background_fill_secondary": "#faf9fc",
        "body_text_color": "#26272e",
        "body_text_color_subdued": "#737580",
        "border_color_primary": "#e9e7ee",
        "block_background_fill": "#ffffff",
        "block_border_color": "#e9e7ee",
        "block_label_background_fill": "#ffffff",
        "block_label_text_color": "#737580",
        "input_background_fill": "#ffffff",
        "input_border_color": "#e9e7ee",
        "input_border_color_focus": "#a78bce",
        "button_primary_background_fill": "#8060a6",
        "button_primary_background_fill_hover": "#6e4e94",
        "button_primary_border_color": "#8060a6",
        "button_primary_text_color": "#ffffff",
        "checkbox_background_color_selected": "#8060a6",
        "checkbox_border_color_selected": "#8060a6",
        "checkbox_label_background_fill_selected": "#f5f0fb",
        "checkbox_label_border_color_selected": "#d7c8eb",
        "slider_color": "#a78bce",
        "loader_color": "#8060a6",
    }
    theme = gr.themes.Base(
        primary_hue="purple",
        neutral_hue="gray",
    ).set(**palette)
    values = theme.to_dict()["theme"]
    return theme.set(**{
        f"{name}_dark": value for name, value in values.items()
        if not name.endswith("_dark") and f"{name}_dark" in values
    })


def integer(value, name, minimum, maximum):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value != int(value)
        or not minimum <= value <= maximum
    ):
        raise gr.Error(f"{name} must be an integer from {minimum} to {maximum}.")
    return int(value)


def prune_outputs():
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    cutoff = time.time() - RETENTION_SECONDS
    for directory in OUTPUTS.iterdir():
        if RUN_ID.fullmatch(directory.name) and directory.is_dir():
            try:
                if directory.stat().st_mtime < cutoff:
                    shutil.rmtree(directory)
            except OSError:
                LOGGER.warning("Could not expire %s", directory.name, exc_info=True)


@lru_cache(maxsize=1)
def get_model():
    from pi4 import Pi4

    return Pi4.from_pretrained(
        os.environ.get("PI4_MODEL_ID", "yyfz233/Pi4"),
        revision=os.environ.get("PI4_MODEL_REVISION") or None,
    )


@spaces.GPU(duration=120)
def predict(frames, query_frame, model):
    if not torch.cuda.is_available():
        raise gr.Error("A CUDA GPU is required. Select ZeroGPU in the Space settings.")
    images = None
    prediction = None
    try:
        model.cuda()
        images = (
            torch.from_numpy(frames.copy())
            .permute(0, 3, 1, 2)
            .float()
            .div_(255)
            .unsqueeze(0)
            .cuda()
        )
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            prediction = model(images, query_frame=query_frame)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        result = {
            name: value.detach().cpu() if isinstance(value, torch.Tensor) else value
            for name, value in prediction.items()
        }
        return result, elapsed
    except torch.cuda.OutOfMemoryError as error:
        raise gr.Error(
            "GPU memory is full. Reduce the frame limit and retry. "
            "The demo does not silently split the sequence or lower its resolution."
        ) from error
    finally:
        model.cpu()
        del images, prediction
        gc.collect()
        torch.cuda.empty_cache()


def reconstruct(
    input_type,
    video,
    images,
    tracking,
    query_frame,
    stride,
    max_frames,
    progress=gr.Progress(),
):
    from pi4.io import load_images
    from pi4.visualization import save_visualization

    stride = integer(stride, "Sampling stride", 1, 60)
    max_frames = integer(max_frames, "Frame limit", 1, 150)
    query = integer(query_frame, "Query frame", 0, 149) if tracking else None
    if input_type == "Video":
        if not video or not Path(video).is_file():
            raise gr.Error("Upload a video or select an example first.")
    elif input_type == "Images":
        if not images or len(images) > 300:
            raise gr.Error("Upload between 1 and 300 images.")
        if any(
            not Path(image).is_file() or Path(image).suffix.lower() not in IMAGE_SUFFIXES
            for image in images
        ):
            raise gr.Error("Images must be JPG, PNG, WEBP, or BMP files.")
    else:
        raise gr.Error("Choose Video or Images as the input type.")

    prune_outputs()
    run = OUTPUTS / uuid.uuid4().hex
    run.mkdir()
    try:
        progress(0.05, desc="Loading and sampling RGB frames on CPU")
        if input_type == "Images":
            source = run / "input"
            source.mkdir()
            for index, image in enumerate(sorted(images, key=lambda item: Path(item).name)):
                image = Path(image)
                shutil.copyfile(image, source / f"{index:06d}{image.suffix.lower()}")
        else:
            source = Path(video)
        frames, metadata = load_images(
            source, stride=stride, max_frames=max_frames, max_size=512
        )
        metadata["source"] = "uploaded images" if input_type == "Images" else source.name
        if input_type == "Images":
            shutil.rmtree(source)
        if tracking and len(frames) < 2:
            raise gr.Error("Tracking needs at least two sampled frames. Disable tracking for one image.")
        if query is not None and query >= len(frames):
            raise gr.Error(
                f"Only {len(frames)} frames were sampled. Query frame must be from 0 "
                f"to {len(frames) - 1}, after sampling."
            )
        progress(0.2, desc="Loading cached weights on CPU; first use may take longer")
        model = get_model()
        progress(0.35, desc="Waiting for ZeroGPU and running one joint forward pass")
        prediction, elapsed = predict(frames, query, model)
        progress(0.8, desc="Exporting the interactive viewer and downloads on CPU")
        data = run / "data"
        scene = save_visualization(data, frames, prediction, metadata, viewer=True)
        del prediction
        archive = shutil.make_archive(str(run / "pi4-result"), "zip", root_dir=data)
        width, height = scene["image_size"]
        status = (
            f"**Done:** {len(frames)} frames · {width} × {height} · "
            f"{elapsed:.1f} s model forward · {scene['preview']['points']:,} preview points.\n\n"
            "One joint forward pass · Results kept for up to 6 hours."
        )
        url = f"/view/{run.name}/"
        viewer = (
            f'<div class="viewer-link"><a href="{url}" target="_blank" rel="noopener">Open in a new tab ↗</a></div>'
            f'<iframe src="{url}" title="Pi4 geometry and motion viewer" '
            'style="width:100%;height:720px;border:0;display:block" '
            'allow="fullscreen" allowfullscreen></iframe>'
        )
        gallery = [
            (frame, f"Source frame {metadata['source_frame_indices'][index]}")
            for index, frame in enumerate(frames[:24])
        ]
        progress(1, desc="Ready")
        return viewer, status, [str(data / scene["preview"]["file"]), archive], gallery
    except gr.Error:
        shutil.rmtree(run, ignore_errors=True)
        raise
    except Exception as error:
        shutil.rmtree(run, ignore_errors=True)
        LOGGER.exception("Pi4 reconstruction failed")
        raise gr.Error(
            "Reconstruction failed. Check that the upload is readable; try fewer frames "
            "to reduce memory use. The Space logs contain the error details."
        ) from error


def load_example(filename):
    stride = 2 if filename == "kart.mp4" else 1
    return "Video", str(ROOT / "examples" / filename), None, True, 0, stride, 150


def create_demo():
    with gr.Blocks(title="π⁴: A Foundation Model for 4D Scene Geometry and Motion", delete_cache=(3600, RETENTION_SECONDS)) as demo:
        gr.HTML(HEADER, elem_id="pi4-header")
        with gr.Column(elem_id="pi4-workspace"):
            with gr.Column(min_width=0, elem_id="pi4-results"):
                with gr.Column(elem_id="pi4-scene-card"):
                    gr.HTML('<div class="card-heading"><span class="section-index">01</span><h2>Scene viewer</h2><span class="card-meta">GEOMETRY · MOTION</span></div>')
                    viewer = gr.HTML(EMPTY_VIEWER, elem_id="pi4-viewer")
                    status = gr.Markdown("Your reconstruction will appear here.", elem_id="pi4-status")
                with gr.Accordion("Downloads & sampled frames", open=False, elem_id="pi4-artifacts"):
                    downloads = gr.File(label="Preview PLY / complete scene ZIP", file_count="multiple", interactive=False)
                    gallery = gr.Gallery(label="Sampled RGB frames (first 24)", columns=6, height=240)
            with gr.Row(elem_id="pi4-controls", equal_height=False):
                with gr.Column(scale=1, min_width=340, elem_id="pi4-input-card"):
                    gr.HTML('<div class="card-heading"><span class="section-index">02</span><h2>Your input</h2><span class="card-meta">RGB ONLY</span></div>')
                    input_type = gr.Radio(
                        ["Video", "Images"], value="Video", label="Input type",
                        show_label=False, elem_id="pi4-input-type",
                    )
                    video = gr.Video(
                        label="Upload a video", sources=["upload"], height=206,
                        show_label=False, elem_id="pi4-video",
                    )
                    images = gr.File(
                        label="Upload images (filename order)",
                        file_count="multiple",
                        file_types=sorted(IMAGE_SUFFIXES),
                        type="filepath",
                        height=206, visible=False, elem_id="pi4-images",
                    )
                    gr.Markdown("Drop media here, or load a clip from the example cards.", elem_id="pi4-upload-hint")
                    tracking = gr.Checkbox(
                        value=True, label="Enable dense motion tracking",
                        elem_id="pi4-tracking",
                    )
                    with gr.Accordion("Sampling settings", open=False, elem_id="pi4-sampling"):
                        stride = gr.Slider(1, 60, value=1, step=1, label="Take every Nth source frame")
                        max_frames = gr.Slider(1, 150, value=150, step=1, label="Maximum sampled frames")
                        query_frame = gr.Number(value=0, precision=0, label="Tracking query frame (after sampling)")
                        gr.Markdown("All sampled frames run together. Lower the frame limit for less GPU memory use. Disable tracking for a single image.")
                    gr.HTML('<div class="inference-defaults"><span>Defaults</span><i></i><span>512 px</span><i></i><span>150 frames</span><i></i><span>Stride 1</span></div>')
                    submit = gr.Button("Reconstruct & Track", variant="primary", elem_id="pi4-submit")
                    gr.Markdown(
                        "Figure Skating defaults to its first 150 frames (~12 s).\n\n"
                        "GPU availability follows your Hugging Face quota.",
                        elem_id="pi4-input-note",
                    )
                with gr.Column(scale=1, min_width=340, elem_id="pi4-examples-card"):
                    gr.HTML('<div class="card-heading"><span class="section-index">03</span><h2>Try an example</h2><span class="card-meta">READY TO EXPLORE</span></div>')
                    gr.Markdown("Two scenes, two kinds of motion. Pick a cover to load its RGB video.", elem_id="pi4-examples-hint")
                    with gr.Row(elem_id="pi4-examples", equal_height=True):
                        for key, title, description, filename, duration in EXAMPLES:
                            card = gr.HTML(
                                f'<button type="button" class="example-card" aria-label="Load {title} example">'
                                f'<span class="example-cover"><img src="{EXAMPLE_COVERS[key]}" alt="{title} video preview" width="512" height="288"/>'
                                '<span class="example-badge">RGB VIDEO</span><span class="example-load" aria-hidden="true">↗</span></span>'
                                f'<span class="example-copy"><span class="example-title">{title}</span>'
                                f'<span class="example-description">{description}</span>'
                                f'<span class="example-meta">{duration}<span>Load clip ↗</span></span></span></button>',
                                js_on_load="element.querySelector('button').addEventListener('click', () => trigger('click'));",
                                elem_id=f"pi4-example-{key}", min_width=0,
                            )
                            card.click(
                                partial(load_example, filename),
                                outputs=[input_type, video, images, tracking, query_frame, stride, max_frames],
                                queue=False, api_visibility="private",
                            )
                    gr.Markdown("Selecting a cover only loads the input. Click **Reconstruct & Track** to run Pi4, then explore the full-width viewer above.", elem_id="pi4-examples-note")
        input_type.change(
            lambda selected: [gr.update(visible=selected == "Video"), gr.update(visible=selected == "Images")],
            inputs=input_type, outputs=[video, images], queue=False, api_visibility="private",
        )
        submit.click(
            reconstruct,
            inputs=[input_type, video, images, tracking, query_frame, stride, max_frames],
            outputs=[viewer, status, downloads, gallery],
            concurrency_limit=1,
            api_name="reconstruct",
        )
        gr.Markdown(
            "**π⁴ · Research demo** &nbsp; CC BY-NC 4.0 · Noncommercial use only.\n\n"
            "Uploads and outputs are temporary, not private storage. Do not upload sensitive content. "
            "Forward timing excludes queueing, model transfer, and export. Third-party terms apply; see LICENSE, LICENSE_MODEL, and NOTICE.",
            elem_id="pi4-footer",
        )
    return demo.queue(max_size=10, default_concurrency_limit=1)


def result_directory(run_id):
    if not RUN_ID.fullmatch(run_id):
        raise HTTPException(status_code=404)
    data = OUTPUTS / run_id / "data"
    if not (data / "scene.json").is_file():
        raise HTTPException(status_code=404)
    return data


def serve_asset(root, asset, suffixes):
    path = (root / asset).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file() or path.suffix not in suffixes:
        raise HTTPException(status_code=404)
    return FileResponse(path, headers={"Cache-Control": "private, no-cache", "X-Content-Type-Options": "nosniff"})


def create_viewer_router():
    router = APIRouter()

    @router.api_route("/view/{run_id}/data/{asset:path}", methods=["GET", "HEAD"])
    def scene_asset(run_id: str, asset: str):
        return serve_asset(result_directory(run_id), asset, {".json", ".bin", ".jpg"})

    @router.api_route("/view/{run_id}/", methods=["GET", "HEAD"])
    def viewer_index(run_id: str):
        result_directory(run_id)
        return serve_asset(VIEWER, "index.html", {".html"})

    @router.api_route("/view/{run_id}/{asset:path}", methods=["GET", "HEAD"])
    def viewer_asset(run_id: str, asset: str):
        result_directory(run_id)
        return serve_asset(VIEWER, asset, {".html", ".js", ".css"})

    return router


demo = create_demo()


def main():
    prune_outputs()
    return demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        share=False,
        app_kwargs={
            "routes": create_viewer_router().routes,
            "middleware": [Middleware(GZipMiddleware, minimum_size=1000)],
        },
        allowed_paths=[str(OUTPUTS), *[str(ROOT / "examples" / example[3]) for example in EXAMPLES]],
        max_file_size="256mb",
        show_error=False,
        ssr_mode=False,
        run_history=False,
        theme=create_theme(),
        css=STYLE,
        footer_links=[],
    )


if __name__ == "__main__":
    main()
