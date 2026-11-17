import base64
import json
from pathlib import Path

import cv2


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def build_covers():
    covers = {}
    for key, filename, frame_index in (("figure_skating", "figure_skating.mp4", 30), ("kart", "kart.mp4", 30)):
        video = cv2.VideoCapture(str(ROOT / "examples" / filename))
        try:
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            readable, frame = video.read()
            if not readable:
                raise RuntimeError(f"Could not decode cover for {filename}")
        finally:
            video.release()
        frame = cv2.resize(frame, (512, 288), interpolation=cv2.INTER_AREA)
        encoded, image = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not encoded:
            raise RuntimeError(f"Could not encode cover for {filename}")
        covers[key] = "data:image/jpeg;base64," + base64.b64encode(image).decode("ascii")
    return json.dumps(covers, indent=2) + "\n"


if __name__ == "__main__":
    print(build_covers(), end="")
