"""Draws tracks onto a frame and encodes it for the browser."""

from __future__ import annotations

import threading

import cv2
import numpy as np

from livedetect.types import Result

_LIVE = (60, 200, 60)  # BGR green: seen in this frame
_COASTING = (160, 160, 160)  # grey: position is predicted
_MIN_BOX = 14  # pixels; a 5-pixel target needs a box the eye can find
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw(result: Result) -> np.ndarray:
    image = result.frame.image.copy()
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


def encode_jpeg(image: np.ndarray, quality: int = 80) -> bytes:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return encoded.tobytes()


class JpegCache:
    """Encodes each result once, however many browsers are watching."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._seq = -1
        self._jpeg = b""

    def get(self, result: Result) -> bytes:
        with self._lock:
            if result.frame.seq != self._seq:
                self._jpeg = encode_jpeg(draw(result))
                self._seq = result.frame.seq
            return self._jpeg
