import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent


def prepare(destination):
    destination = destination.resolve()
    if not destination.is_relative_to((ROOT / "outputs").resolve()):
        raise ValueError("Prepare the Space inside this project's outputs/ directory.")
    if destination.exists():
        raise FileExistsError(f"Choose a new output directory: {destination}")
    destination.mkdir(parents=True)
    for directory in ("pi4", "viewer", "examples", "licenses"):
        shutil.copytree(
            ROOT / directory,
            destination / directory,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )
    for filename in ("pyproject.toml", "LICENSE", "LICENSE_MODEL", "NOTICE"):
        shutil.copy2(ROOT / filename, destination / filename)
    for filename in ("app.py", "space.css", "example-covers.json", "requirements.txt", "README.md", ".gitattributes"):
        shutil.copy2(HERE / filename, destination / filename)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    )
    files = {
        str(path.relative_to(destination)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(destination.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "note": "Hashes describe the working-tree snapshot, including uncommitted edits.",
        "sha256": files,
    }
    (destination / "deployment-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return destination


def main():
    parser = argparse.ArgumentParser(description="Prepare and optionally upload the Pi4 ZeroGPU Space")
    parser.add_argument("--repo-id", default="yyfz233/Pi4")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--upload", action="store_true", help="Upload using the existing Hugging Face login")
    arguments = parser.parse_args()
    destination = arguments.output or ROOT / "outputs" / (
        "huggingface-space-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    )
    destination = prepare(destination)
    print(f"Space snapshot: {destination}", flush=True)
    if arguments.upload:
        from huggingface_hub import HfApi
        from huggingface_hub.errors import HfHubHTTPError

        api = HfApi()
        try:
            space = api.space_info(arguments.repo_id)
            result = api.upload_folder(
                repo_id=arguments.repo_id,
                repo_type="space",
                folder_path=str(destination),
                commit_message="Add Pi4 ZeroGPU geometry and motion demo",
                parent_commit=space.sha,
            )
        except HfHubHTTPError as error:
            status = error.response.status_code if error.response is not None else "unknown"
            raise SystemExit(
                f"Upload failed (HTTP {status}). The local snapshot is preserved at {destination}. "
                "Check that the Space exists and the logged-in token can read and write it."
            ) from None
        print(f"Uploaded: {result}")


if __name__ == "__main__":
    main()
