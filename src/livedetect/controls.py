"""The switches on the live page: faults to inject and modes to compare.

    detector_delay_ms  slow the detector down by this much per frame
    camera_cut         make the source fail until switched back
    queue              hand frames over through a queue instead of dropping
    tiles              run the detector on a grid of tiles, e.g. "3x2"

A button that cuts the camera does not belong on a service that is watching
something, so none of this exists unless the service is started with
--controls.
"""

from __future__ import annotations

from livedetect.detectors import TiledDetector, parse_grid
from livedetect.faults import Faults
from livedetect.pipeline import Pipeline

MAX_DELAY_MS = 2000


class Controls:
    def __init__(self, pipeline: Pipeline, faults: Faults, tiled: TiledDetector | None) -> None:
        self.pipeline, self.faults, self.tiled = pipeline, faults, tiled

    def state(self) -> dict:
        return {
            "detector_delay_ms": round(self.faults.detector_delay_s * 1e3),
            "camera_cut": self.faults.camera_cut,
            "queue": self.pipeline.queueing,
            # None: this detector cannot be tiled.
            "tiles": None if self.tiled is None else "{}x{}".format(*self.tiled.grid),
        }

    def apply(self, changes: dict) -> None:
        """Change the named switches. Raises ValueError and changes nothing if any is invalid."""
        new = {name: self._checked(name, value) for name, value in changes.items()}
        if "detector_delay_ms" in new:
            self.faults.detector_delay_s = new["detector_delay_ms"] / 1e3
        if "camera_cut" in new:
            self.faults.camera_cut = new["camera_cut"]
        if "queue" in new:
            self.pipeline.set_queueing(new["queue"])
        if "tiles" in new:
            self.tiled.grid = new["tiles"]

    def _checked(self, name: str, value: object) -> object:
        if name == "detector_delay_ms":
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError("detector_delay_ms must be a number")
            if not 0 <= value <= MAX_DELAY_MS:
                raise ValueError(f"detector_delay_ms must be between 0 and {MAX_DELAY_MS}")
            return value
        if name in ("camera_cut", "queue"):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be true or false")
            return value
        if name == "tiles":
            if self.tiled is None:
                raise ValueError("this detector cannot be tiled")
            if not isinstance(value, str):
                raise ValueError("tiles must be text like 3x2")
            return parse_grid(value)
        raise ValueError(f"unknown control {name!r}")
