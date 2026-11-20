"""Fakes and small utilities shared by the tests."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from livedetect.sources import SourceError
from livedetect.types import Detection


def wait_until(predicate: Callable[[], bool], timeout: float = 5.0) -> bool:
    """Poll until `predicate()` is true. Avoids fixed sleeps in threaded tests."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.002)
    return False


class FakeSource:
    """Delivers blank frames at a fixed rate and fails when told to.

    frames_per_connection  the stream breaks after this many frames (None: never)
    failing_opens          the first N calls to open() fail
    """

    def __init__(
        self,
        period_s: float = 0.002,
        frames_per_connection: int | None = None,
        failing_opens: int = 0,
        error: type[SourceError] = SourceError,
    ) -> None:
        self.period_s = period_s
        self.frames_per_connection = frames_per_connection
        self.failing_opens = failing_opens
        self.error = error
        self.opens = 0
        self.closes = 0
        self._left: int | None = None

    def open(self) -> None:
        self.opens += 1
        if self.opens <= self.failing_opens:
            raise SourceError("camera unreachable")
        self._left = self.frames_per_connection

    def read(self) -> np.ndarray:
        time.sleep(self.period_s)
        if self._left is not None:
            if self._left == 0:
                raise self.error("stream broke")
            self._left -= 1
        return np.zeros((8, 8, 3), np.uint8)

    def close(self) -> None:
        self.closes += 1


class FakeDetector:
    """Takes `delay_s` per frame and returns one fixed detection."""

    def __init__(self, delay_s: float = 0.0, fail_every: int | None = None) -> None:
        self.delay_s = delay_s
        self.fail_every = fail_every
        self.calls = 0

    def detect(self, image: np.ndarray) -> list[Detection]:
        self.calls += 1
        if self.fail_every and self.calls % self.fail_every == 0:
            raise RuntimeError("model crashed")
        time.sleep(self.delay_s)
        return [Detection(4.0, 4.0, 2.0, 2.0, 0.9)]
