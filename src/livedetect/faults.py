"""Faults that can be switched on while the service runs.

A design that claims to survive a slow detector and a lost camera should be
shown doing it. These wrappers sit in front of the real source and detector
and misbehave on request. They are only installed when the service is started
with --controls.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from livedetect.detectors import Detector
from livedetect.sources import FrameSource, SourceError
from livedetect.types import Detection


@dataclass
class Faults:
    """What is currently broken. Written by an HTTP thread, read by the pipeline's.

    Each field is a single value replaced in one step, so no lock is needed.
    """

    detector_delay_s: float = 0.0  # added to every detection
    camera_cut: bool = False  # the source fails as if the cable were pulled


class FaultySource:
    """A source that can be cut off."""

    def __init__(self, inner: FrameSource, faults: Faults) -> None:
        self.inner, self.faults = inner, faults

    def open(self) -> None:
        self._fail_if_cut()
        self.inner.open()

    def read(self) -> np.ndarray:
        self._fail_if_cut()
        return self.inner.read()

    def close(self) -> None:
        self.inner.close()

    def _fail_if_cut(self) -> None:
        if self.faults.camera_cut:
            raise SourceError("camera cut by the fault switch")


class FaultyDetector:
    """A detector that can be slowed down."""

    def __init__(self, inner: Detector, faults: Faults) -> None:
        self.inner, self.faults = inner, faults

    def detect(self, image: np.ndarray) -> list[Detection]:
        delay = self.faults.detector_delay_s
        if delay > 0:
            time.sleep(delay)
        return self.inner.detect(image)
