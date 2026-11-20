"""How small can the drone be before the detector loses it, and what does tiling buy?

    python bench/tiny_targets.py assets/drone_far.mp4

There is no labelled footage of tiny drones in this repository, so this makes
its own from a clip where the detector is reliable:

1. Run the plain detector on the original frames. Where it is confident, its
   box is the reference: this is where the drone is.
2. Paint the drone out of the frame, shrink it by a factor k, and paste it back
   at the same place. The scene is unchanged; the drone looks k times further
   away.
3. Run each detector on the altered frames and count how often it reports a
   box on the drone, and how often it reports a box anywhere else.

Painting the drone out only works against open sky, so frames where it is in
front of anything else are left out. What this does not capture: haze and loss
of contrast with distance, motion blur, and clutter behind the target. Real
range is harder than this.
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from livedetect.detectors import OnnxYoloDetector, TiledDetector, parse_grid
from livedetect.models import fetch
from livedetect.types import Detection

REFERENCE_SCORE = 0.6  # the plain detector must be at least this sure of the original
MATCH_SLACK_PX = 6.0  # a 5-pixel target cannot be boxed to the pixel
SKY_ROUGHNESS = 1.5  # grey levels; open sky measures under 1, foliage over 2


def patch_corners(image: np.ndarray, box: Detection) -> tuple[int, int, int, int]:
    """The box and as much again of its surroundings, kept inside the frame."""
    height, width = image.shape[:2]
    x0, x1 = int(max(0, box.cx - box.w)), int(min(width, box.cx + box.w))
    y0, y1 = int(max(0, box.cy - box.h)), int(min(height, box.cy + box.h))
    return x0, y0, x1, y1


def against_sky(image: np.ndarray, box: Detection) -> bool:
    """True if the surroundings of the box are smooth enough to paint the drone out.

    Measured as the mean step between neighbouring pixels along the edge of
    the patch: small for sky and cloud, several times larger for trees.
    """
    x0, y0, x1, y1 = patch_corners(image, box)
    grey = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    edge = np.concatenate([grey[0], grey[-1], grey[:, 0], grey[:, -1]])
    return float(np.abs(np.diff(edge)).mean()) < SKY_ROUGHNESS


def shrink_target(image: np.ndarray, box: Detection, k: float) -> np.ndarray:
    """Return a copy of `image` with the target in `box` made `k` times smaller."""
    if k == 1:
        return image
    # The edge of the patch is sky on sky, so the result blends without a seam.
    x0, y0, x1, y1 = patch_corners(image, box)
    patch = image[y0:y1, x0:x1]
    plate = image.copy()
    plate[y0:y1, x0:x1] = fill_from_edges(patch)  # the scene without the drone

    small_w, small_h = max(2, round(patch.shape[1] / k)), max(2, round(patch.shape[0] / k))
    small = cv2.resize(patch, (small_w, small_h), interpolation=cv2.INTER_AREA)
    alpha = np.minimum.outer(_fade(small_h), _fade(small_w))[:, :, None]
    # Shrink towards the centre of the box, so the drone stays where it was.
    left = round(box.cx - (box.cx - x0) / k)
    top = round(box.cy - (box.cy - y0) / k)
    region = plate[top : top + small_h, left : left + small_w]
    region[:] = (alpha * small + (1 - alpha) * region).astype(np.uint8)
    return plate


def _fade(n: int) -> np.ndarray:
    """n weights: 1 in the middle, falling to 0 at both ends over a fifth of the length."""
    to_nearer_end = np.minimum(np.arange(n) + 0.5, n - 0.5 - np.arange(n))
    return np.clip(to_nearer_end / (0.2 * n), 0, 1)


def fill_from_edges(patch: np.ndarray) -> np.ndarray:
    """A smooth surface through the four edges of `patch`: sky without the drone.

    Each pixel is a blend of the edge pixels in its row and column (a Coons
    patch). Good for sky and cloud; it would smear anything with texture.
    """
    edges = patch.astype(np.float32)
    h, w = edges.shape[:2]
    u = np.linspace(0, 1, w, dtype=np.float32)[None, :, None]
    v = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    top, bottom, left, right = edges[:1], edges[-1:], edges[:, :1], edges[:, -1:]
    across = (1 - u) * left + u * right
    down = (1 - v) * top + v * bottom
    corners = (
        (1 - u) * (1 - v) * edges[0, 0] + u * (1 - v) * edges[0, -1]
        + (1 - u) * v * edges[-1, 0] + u * v * edges[-1, -1]
    )  # fmt: skip
    return np.clip(across + down - corners, 0, 255).astype(np.uint8)


def on_target(d: Detection, box: Detection, k: float) -> bool:
    return abs(d.cx - box.cx) <= max(box.w / k / 2, MATCH_SLACK_PX) and abs(d.cy - box.cy) <= max(
        box.h / k / 2, MATCH_SLACK_PX
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("clip")
    parser.add_argument("--grids", default="1x1,3x2,5x3", help="tile grids to compare")
    parser.add_argument("--shrink", default="1,1.5,2,3,4,5,6,8", help="factors k")
    parser.add_argument("--every", type=int, default=3, help="use every Nth frame")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads for the network")
    parser.add_argument("--example", help="write one altered frame per factor to this image file")
    args = parser.parse_args()
    grids = [parse_grid(g) for g in args.grids.split(",")]
    factors = [float(k) for k in args.shrink.split(",")]

    network = OnnxYoloDetector(fetch("drone"), threads=args.threads)
    video = cv2.VideoCapture(args.clip)
    frames: list[tuple[np.ndarray, Detection]] = []
    index = 0
    while True:
        ok, image = video.read()
        if not ok:
            break
        if index % args.every == 0:
            found = network.detect(image)
            sure = len(found) == 1 and found[0].score >= REFERENCE_SCORE
            if sure and against_sky(image, found[0]):
                frames.append((image, found[0]))
        index += 1
    widths = [box.w for _, box in frames]
    print(
        f"{args.clip}: {len(frames)} frames with the drone against sky, "
        f"where it is {np.median(widths):.0f} px wide\n"
    )

    hits = {(g, k): 0 for g in grids for k in factors}
    false_alarms = dict.fromkeys(hits, 0)
    seconds = dict.fromkeys(grids, 0.0)
    examples = []
    for n, (image, box) in enumerate(frames):
        for k in factors:
            altered = shrink_target(image, box, k)
            if n == len(frames) // 2 and args.example:
                examples.append(
                    altered[
                        int(box.cy) - 60 : int(box.cy) + 60, int(box.cx) - 100 : int(box.cx) + 100
                    ]
                )
            for g in grids:
                started = time.perf_counter()
                found = TiledDetector(network, *g).detect(altered)
                seconds[g] += time.perf_counter() - started
                good = [d for d in found if on_target(d, box, k)]
                hits[g, k] += bool(good)
                false_alarms[g, k] += len(found) - len(good)
    if args.example:
        cv2.imwrite(args.example, np.hstack(examples))

    runs = len(frames) * len(factors)
    names = [f"{c}x{r}" if (c, r) != (1, 1) else "plain" for c, r in grids]
    print("Frames in which the drone was found\n")
    print("| Drone width (px) | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(grids))
    for k in factors:
        cells = [f"{hits[g, k] / len(frames):.0%}" for g in grids]
        print(f"| {np.median(widths) / k:.0f} | " + " | ".join(cells) + " |")
    print("\n| | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(grids))
    print(
        "| Network runs per frame | "
        + " | ".join(str(c * r + (c * r > 1)) for c, r in grids)
        + " |"
    )
    print(
        f"| Time per frame (ms, {args.threads} threads) | "
        + " | ".join(f"{seconds[g] / runs * 1e3:.0f}" for g in grids)
        + " |"
    )
    alarms = [sum(false_alarms[g, k] for k in factors) / runs * 100 for g in grids]
    print("| False alarms per 100 frames | " + " | ".join(f"{a:.1f}" for a in alarms) + " |")


if __name__ == "__main__":
    main()
