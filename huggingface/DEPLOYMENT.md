# Deploying the Pi4 ZeroGPU Space

The files in this directory are project-owned deployment sources. The deployment
helper assembles a self-contained Space under `outputs/`, copying the current
Pi4 code, native viewer, examples, dependencies, and licenses. It does not edit
the main repository's README, commit the source repository, or copy credentials,
checkpoints, inference outputs, or scheduler temporary files.

## Prepare or Upload

From the Pi4 repository root:

```bash
python huggingface/deploy.py --output outputs/huggingface-space
```

Or prepare a new snapshot and upload directly using the current Hugging Face
login:

```bash
python huggingface/deploy.py --upload --repo-id yyfz233/Pi4
```

Upload requires `huggingface_hub` in the active Python environment, an existing
Space, and a token with read/write access to that Space. For a fine-grained token,
explicitly grant access to the organization and the target repository. A personal
login or organization membership alone does not establish the token's scope.
The helper does not create repositories or change the organization's subscription
or hardware settings. If upload fails, the complete snapshot stays on disk.

When publishing through a Git checkout instead, install Git LFS and run
`git lfs install --local` before staging the Space files. The prepared
`.gitattributes` includes MP4 tracking; `git lfs ls-files` must show all three bundled
videos before the first push. Hugging Face supports Git LFS through its Xet/LFS
bridge. If the videos were already included as ordinary blobs in a single
rejected, unpublished commit, renormalize them with
`git add --renormalize examples/figure_skating.mp4 examples/kart.mp4 examples/cycling.mp4`, stage
`.gitattributes`, and amend that commit. Merely adding a follow-up commit leaves
the rejected binary blobs in the history being pushed.

To upload an already prepared snapshot later:

```bash
hf upload yyfz233/Pi4 outputs/huggingface-space . --repo-type space
```

In the Space settings:

- Select the **Gradio** SDK and **ZeroGPU** hardware.
- Confirm that the owning organization is eligible to host ZeroGPU Spaces.
- Keep `app.py` as the application entrypoint and Python 3.12.
- For private model weights, add an `HF_TOKEN` Secret with read access to the
  model repository. A local `hf auth login` does not authenticate the Space.
- Do not replace the torch pin with Pi3's older requirements or force-install
  conflicting packages using `--no-deps`.
- Run a short example after the build finishes to validate the actual ZeroGPU
  CUDA environment, Triton compilation, and FlexGEMM sparse refinement.

The source's GPU functions do not use `torch.compile`; the sparse backend uses
Triton kernels. Whether those kernels run on the current hosted GPU must be
checked on the Space itself, not inferred from a local GPU smoke test.

The README intentionally omits `preload_from_hub`, which does not support
private models. Weights are downloaded at runtime on the first request using
`HF_TOKEN`, before GPU allocation. Never commit the token or store it in a
public Space Variable.

## ZeroGPU Startup

The entrypoint uses Gradio `launch()` to execute the Spaces SDK's ZeroGPU
startup hook. Do not replace it with a direct Uvicorn server around a mounted
Gradio app: that skips startup reporting and can cause
`No @spaces.GPU function detected during startup`. Viewer routes are passed
through `launch(app_kwargs=...)` so they share the same server.

The launched Blocks instance is stored in the module-level `demo` variable so
the Spaces hot reloader can find it. Keep this instance at module scope rather
than constructing it only inside `main()`.

## Demo Defaults and Viewer

The Gradio shell uses the local `space.css` file and a light theme supplied to
`launch()`. Keep `space.css` alongside `app.py` when preparing or updating the
Space; `deploy.py` includes it in snapshots. Keep `example-covers.json` alongside
them as well: it contains small JPEG data URLs extracted from the bundled videos,
so the example covers need no external image host or new binary Git blobs.
Regenerate it in the source repository with
`python huggingface/build_example_covers.py > huggingface/example-covers.json`.
The viewer occupies the full width; uploads and clickable example covers sit
below it. The header links to the official project page. Outer-page styling is
separate from the embedded viewer, which retains its own controls and appearance.
Keep Gradio's default font metadata in the theme and customize the font stack
in CSS. Gradio 6.31's launch analytics compares custom themes with built-ins;
a plain-string font list can raise `AttributeError` in `Font.__eq__` and abort
startup. Validate startup with analytics enabled as well as disabled.

The UI defaults use stride 1, at most 150 frames, query frame 0,
and a fixed 512-pixel longest side, matching the repository quickstart. Figure Skating
therefore uses its first 150 consecutive frames, not a strided 50-frame clip.
The CLI defaults to no frame cap; the hosted Space retains its 150-frame cap
to bound GPU use.
Selecting the Kart example sets stride 2 and 150 frames to cover its full clip.
The repository also bundles Cycling; its full stride-2 command processes 178 frames,
documented in the main README and exceeds this Space's 150-frame limit.
The reconstruction API takes seven inputs; it no longer accepts `max_size`.

`viewer/static-demo-viewer.js` prefetches 32 future frames and limits bulk
downloads to three, sharing those slots with the asynchronous track download.
Chunks are deduplicated, prioritized for the selected frame, and applied in
cooperative batches. A failed speculative request is retried when needed, not
in a background loop. `viewer/frame-image-cache.js` keeps up to 29 RGB frames
(24 ahead, current, four behind), with at most three image requests in flight.
Keep that module with the other viewer assets when updating the Space.

The viewer uses a light website-style layout, RGB camera inset, filmstrip,
speed control, camera frustums, and fullscreen. Its render-budget slider defaults
to at most 2,000,000 points, bounded by the exported geometry. Lowering the
slider rebuilds the indexed GPU draw list using deterministic `linspace`;
it does not just discard points in a shader or trigger inference again.
The exporter retains its existing 40 MB, approximately two-million-point cap.

## Run Locally

On a CUDA machine, either run the project-owned entrypoint:

```bash
python -m pip install -r huggingface/requirements.txt
python huggingface/app.py
```

Or enter the prepared Space directory and run:

```bash
python -m pip install -r requirements.txt
python app.py
```

Open port 7860. The `spaces.GPU` decorator is inactive outside ZeroGPU. Do not
expose the local server to untrusted users without the appropriate network
controls. This is a public research demo, not a private upload service.

The original command-line viewer continues to use `/data/`. Its asset URLs are
now relative to the viewer page, so the same viewer also works under the
Space's isolated per-run `/view/<random-id>/` routes.

## Dependency Build Order

The Gradio Space installs `requirements.txt` before copying the application
source into `/app`. Its requirements must therefore list the runtime packages
directly; an editable install such as `-e .[example]` fails because
`pyproject.toml` is not available at that stage. The application imports the
bundled `pi4/` source directly, including when running `huggingface/app.py` from
the source repository. Keep the explicit model dependency pins synchronized
with the root `pyproject.toml`.
