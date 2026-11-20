"""Draws results onto frames and encodes them for the browser.

Two pictures come out of every result: the whole frame with a box on each
track, and the tracking camera's view, which is a part of the frame enlarged.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

import cv2
import numpy as np

from livedetect.types import Result

_LIVE = (60, 200, 60)  # BGR green: seen in this frame
_COASTING = (160, 160, 160)  # grey: position is predicted
_WINDOW = (255, 255, 255)  # white: what the tracking camera is looking at
_MIN_BOX = 14  # pixels; a 5-pixel target needs a box the eye can find
_FONT = cv2.FONT_HERSHEY_SIMPLEX
VIEW_WIDTH = 480  # pixels, the tracking camera's picture


def draw(result: Result) -> np.ndarray:
    """The whole frame, with tracks and the tracking camera's window."""
    image = result.frame.image.copy()
    view = result.view
    if view is not None and view.zoom > 1.05:  # at 1x the window is the frame itself
        x0, y0 = int(view.cx - view.w / 2), int(view.cy - view.h / 2)
        x1, y1 = int(view.cx + view.w / 2), int(view.cy + view.h / 2)
        cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 0), 3)
        cv2.rectangle(image, (x0, y0), (x1, y1), _WINDOW, 1)
    for t in result.tracks:
        colour = _COASTING if t.coasting else _LIVE
        half_w, half_h = max(t.w, _MIN_BOX) / 2, max(t.h, _MIN_BOX) / 2
        x0, y0 = int(t.cx - half_w), int(t.cy - half_h)
        x1, y1 = int(t.cx + half_w), int(t.cy + half_h)
        cv2.rectangle(image, (x0, y0), (x1, y1), colour, 2)
        # Label: dark text on a filled tab, readable against sky and clouds alike.
        text = f"#{t.id} {t.label} {t.score:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(text, _FONT, 0.5, 1)
        tab_top = max(y0 - text_h - 8, 0)
        cv2.rectangle(image, (x0 - 1, tab_top), (x0 + text_w + 6, tab_top + text_h + 8), colour, -1)
        cv2.putText(
            image, text, (x0 + 3, tab_top + text_h + 4), _FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA
        )
    return image


def draw_view(result: Result) -> np.ndarray:
    """The tracking camera's picture: its window of the frame, enlarged."""
    image, view = result.frame.image, result.view
    height, width = image.shape[:2]
    out_w, out_h = VIEW_WIDTH, round(VIEW_WIDTH * height / width / 2) * 2
    if view is None:
        return cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_AREA)

    # One affine map takes frame pixels to view pixels. Resampling through it,
    # instead of cropping whole pixels and resizing, lets the window move by
    # fractions of a pixel, so the enlarged picture glides and does not judder.
    scale = out_w / view.w
    left, top = view.cx - view.w / 2, view.cy - view.h / 2
    to_view = np.float32([
        [scale, 0, scale * (0.5 - left) - 0.5],
        [0, scale, scale * (0.5 - top) - 0.5],
    ])  # fmt: skip
    smooth = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    out = cv2.warpAffine(
        image, to_view, (out_w, out_h), flags=smooth, borderMode=cv2.BORDER_REPLICATE
    )

    status = "SEARCHING"
    for t in result.tracks:
        if t.id != view.target:
            continue
        status = f"{'PREDICTED' if t.coasting else 'LOCKED'} #{t.id}"
        colour = _COASTING if t.coasting else _LIVE
        half_w, half_h = max(t.w, _MIN_BOX) * scale / 2 + 6, max(t.h, _MIN_BOX) * scale / 2 + 6
        cx, cy = (t.cx - left) * scale, (t.cy - top) * scale
        _corner_brackets(out, cx - half_w, cy - half_h, cx + half_w, cy + half_h, colour)
    _caption(out, f"{status}  {view.zoom:.1f}x")
    return out


def _corner_brackets(image: np.ndarray, x0: float, y0: float, x1: float, y1: float, colour) -> None:
    """Four corners instead of a full box: marks the target without covering it."""
    arm = int(max(6, min(x1 - x0, y1 - y0) / 4))
    for x, y, dx, dy in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        corner = (int(x), int(y))
        cv2.line(image, corner, (int(x + dx * arm), int(y)), colour, 2, cv2.LINE_AA)
        cv2.line(image, corner, (int(x), int(y + dy * arm)), colour, 2, cv2.LINE_AA)


def _caption(image: np.ndarray, text: str) -> None:
    (text_w, text_h), _ = cv2.getTextSize(text, _FONT, 0.5, 1)
    cv2.rectangle(image, (0, 0), (text_w + 12, text_h + 12), (0, 0, 0), -1)
    cv2.putText(image, text, (6, text_h + 6), _FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def encode_jpeg(image: np.ndarray, quality: int = 80) -> bytes:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return encoded.tobytes()


class JpegCache:
    """Draws and encodes each result once, however many browsers are watching."""

    def __init__(self, draw: Callable[[Result], np.ndarray] = draw) -> None:
        self._draw = draw
        self._lock = threading.Lock()
        self._seq = -1
        self._jpeg = b""

    def get(self, result: Result) -> bytes:
        with self._lock:
            if result.frame.seq != self._seq:
                self._jpeg = encode_jpeg(self._draw(result))
                self._seq = result.frame.seq
            return self._jpeg
