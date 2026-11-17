import argparse
from functools import lru_cache, partial
import gzip
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import shutil
from urllib.parse import unquote, urlsplit

ASSETS = Path(__file__).resolve().parent


@lru_cache(maxsize=2)
def compressed_asset(path, modified):
    return gzip.compress(path.read_bytes(), compresslevel=1, mtime=0)


class ViewerHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, data, **kwargs):
        self.data = data
        super().__init__(*args, **kwargs)

    def do_GET(self):
        try:
            self.send_asset()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_HEAD(self):
        self.send_asset(head=True)

    def send_asset(self, head=False):
        requested = unquote(urlsplit(self.path).path)
        if requested.startswith("/data/"):
            root, relative = self.data, requested[len("/data/") :]
        else:
            root, relative = ASSETS, requested.lstrip("/") or "index.html"
        path = (root / relative).resolve()
        if (
            not path.is_relative_to(root.resolve())
            or not path.is_file()
            or path.suffix not in {".html", ".css", ".js", ".json", ".bin", ".jpg"}
        ):
            self.send_error(404)
            return
        information = path.stat()
        compressed = path.suffix == ".bin" and "gzip" in {
            value.strip()
            for value in self.headers.get("Accept-Encoding", "").split(",")
        }
        etag = (
            f'"{information.st_mtime_ns:x}-{information.st_size:x}-{int(compressed)}"'
        )
        unchanged = self.headers.get("If-None-Match") == etag
        self.send_response(304 if unchanged else 200)
        self.send_header("ETag", etag)
        self.send_header(
            "Cache-Control",
            "private, no-cache" if requested.startswith("/data/") else "no-store",
        )
        self.send_header("Vary", "Accept-Encoding")
        if unchanged:
            self.end_headers()
            return
        content = (
            compressed_asset(path, information.st_mtime_ns) if compressed else None
        )
        self.send_header(
            "Content-Type",
            mimetypes.guess_type(path.name)[0] or "application/octet-stream",
        )
        self.send_header(
            "Content-Length",
            str(len(content) if content is not None else information.st_size),
        )
        if compressed:
            self.send_header("Content-Encoding", "gzip")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if not head:
            if content is not None:
                self.wfile.write(content)
            else:
                with path.open("rb") as stream:
                    shutil.copyfileobj(stream, self.wfile)


def main():
    parser = argparse.ArgumentParser(
        description="Serve the lightweight Pi4 point-cloud viewer"
    )
    parser.add_argument(
        "data", type=Path, help="output directory exported with example.py --viewer"
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="127.0.0.1")
    arguments = parser.parse_args()
    data = arguments.data.resolve()
    scene = json.loads((data / "scene.json").read_text())
    if scene["format"] != "pi4_scene_v1" or not scene.get("viewer"):
        raise ValueError("Not a Pi4 viewer export")
    with ThreadingHTTPServer(
        (arguments.host, arguments.port), partial(ViewerHandler, data=data)
    ) as server:
        print(f"Pi4 viewer: http://{arguments.host}:{server.server_port}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
