"""Write a copy of a clip in which the drone is smaller: a stand-in for a distant one.

    python bench/make_tiny_clip.py assets/drone_far.mp4 assets/drone_tiny.mp4 --seconds 12.5

Uses the same edit as tiny_targets.py, frame by frame, and like it only works
while the drone is against open sky; `--seconds` cuts the clip before it is
not. Needs ffmpeg.
"""

from __future__ import annotations

import argparse
import subprocess

import cv2
from tiny_targets import REFERENCE_SCORE, against_sky, shrink_target

from livedetect.detectors import OnnxYoloDetector
from livedetect.models import fetch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("clip")
    parser.add_argument("out")
    parser.add_argument("--shrink", type=float, default=6.0)
    parser.add_argument("--seconds", type=float, help="stop after this much of the clip")
    args = parser.parse_args()

    network = OnnxYoloDetector(fetch("drone"))
    video = cv2.VideoCapture(args.clip)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    encoder = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-crf", "18", "-preset", "slow", "-pix_fmt", "yuv420p", args.out],
        stdin=subprocess.PIPE,
    )  # fmt: skip
    box, frames, unsure = None, 0, 0
    while args.seconds is None or frames < args.seconds * fps:
        ok, image = video.read()
        if not ok:
            break
        found = [d for d in network.detect(image) if d.score >= REFERENCE_SCORE]
        if found:
            box = max(found, key=lambda d: d.score)
        else:
            unsure += 1  # keep editing where the drone last was; it moves slowly
        if box is not None:
            if not against_sky(image, box):
                encoder.kill()
                raise SystemExit(
                    f"frame {frames} ({frames / fps:.1f} s): the drone is not against open sky; "
                    "stop earlier with --seconds"
                )
            image = shrink_target(image, box, args.shrink)
        frames += 1
        encoder.stdin.write(image.tobytes())
    encoder.stdin.close()
    encoder.wait()
    print(f"{args.out}: {frames} frames, {unsure} edited at the previous position")


if __name__ == "__main__":
    main()
