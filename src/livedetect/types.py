"""The data that flows between pipeline stages. All of it is immutable."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Frame:
    seq: int  # counts frames since the service started
    captured_at: float  # monotonic seconds, taken when the frame entered this process
    image: np.ndarray  # height x width x 3, uint8, BGR


@dataclass(frozen=True, slots=True)
class Detection:
    cx: float  # box centre, pixels
    cy: float
    w: float
    h: float
    score: float  # 0..1
    label: str = "object"


@dataclass(frozen=True, slots=True)
class Track:
    id: int
    cx: float
    cy: float
    w: float
    h: float
    vx: float  # pixels per second
    vy: float
    score: float
    label: str
    hits: int  # detections matched to this track so far
    coasting: bool  # True while the position is predicted, not observed


@dataclass(frozen=True, slots=True)
class Result:
    frame: Frame
    tracks: tuple[Track, ...]
    n_detections: int
    processing_s: float  # detector plus tracker
    latency_s: float  # frame captured -> result ready

    def to_dict(self) -> dict:
        """JSON-ready summary, without the image."""
        return {
            "seq": self.frame.seq,
            "latency_ms": round(self.latency_s * 1e3, 2),
            "processing_ms": round(self.processing_s * 1e3, 2),
            "detections": self.n_detections,
            "tracks": [
                {k: round(v, 2) if isinstance(v, float) else v for k, v in asdict(t).items()}
                for t in self.tracks
            ],
        }
