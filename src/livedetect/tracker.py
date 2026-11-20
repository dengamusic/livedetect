"""Links detections across frames into tracks with stable ids.

Matching is by distance between box centres, not box overlap: the targets are
a few pixels wide, and two boxes that small rarely overlap from one frame to
the next even when they are the same object.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from livedetect.types import Detection, Track


@dataclass
class _State:
    cx: float
    cy: float
    w: float
    h: float
    score: float
    label: str
    vx: float = 0.0
    vy: float = 0.0
    hits: int = 1
    last_seen: float = 0.0  # time of the last detection matched to this track
    id: int = 0  # 0 until the track is confirmed


class Tracker:
    """Constant-velocity tracker with an alpha-beta filter.

    gate_px       how far a detection may be from a track's predicted position
    gate_px_s     added to the gate per second since the last frame: how fast a
                  target may leave its predicted path (a turn, a shaking camera)
    confirm_hits  detections needed before a track is reported (rejects noise)
    max_coast_s   how long a confirmed track survives without a detection
    alpha, beta   how strongly a detection corrects position and velocity
    max_gap_s     a pause longer than this (a reconnect) discards all tracks

    Limits are in seconds, not frames, because the frame rate is not fixed: it
    falls whenever the detector is slower than the camera.
    """

    def __init__(
        self,
        gate_px: float = 20.0,
        gate_px_s: float = 450.0,
        confirm_hits: int = 3,
        max_coast_s: float = 0.5,
        alpha: float = 0.6,
        beta: float = 0.3,
        max_gap_s: float = 1.0,
    ) -> None:
        self.gate_px, self.gate_px_s = gate_px, gate_px_s
        self.confirm_hits, self.max_coast_s = confirm_hits, max_coast_s
        self.alpha, self.beta, self.max_gap_s = alpha, beta, max_gap_s
        self._tracks: list[_State] = []
        self._last_t: float | None = None
        self._next_id = 1

    def update(self, detections: Sequence[Detection], t: float) -> list[Track]:
        """Advance to time `t` (seconds) and return the confirmed tracks."""
        dt = 0.0 if self._last_t is None else t - self._last_t
        if dt < 0 or dt > self.max_gap_s:
            self._tracks.clear()
            dt = 0.0
        self._last_t = t

        # 1. Predict where each track is now.
        for s in self._tracks:
            s.cx += s.vx * dt
            s.cy += s.vy * dt

        # 2. Match tracks to detections, closest pair first.
        pairs = []
        for ti, s in enumerate(self._tracks):
            gate = self.gate_px + self.gate_px_s * dt + 0.5 * max(s.w, s.h)
            for di, d in enumerate(detections):
                distance = math.hypot(d.cx - s.cx, d.cy - s.cy)
                if distance <= gate:
                    pairs.append((distance, ti, di))
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        for _distance, ti, di in sorted(pairs):
            if ti in matched_tracks or di in matched_detections:
                continue
            matched_tracks.add(ti)
            matched_detections.add(di)
            self._correct(self._tracks[ti], detections[di], t)

        # 3. Tracks with no detection coast on their prediction, then die. An
        #    unconfirmed one dies at once: noise rarely repeats in the same place.
        survivors = []
        for ti, s in enumerate(self._tracks):
            if ti not in matched_tracks:
                confirmed = s.id != 0
                if not confirmed or t - s.last_seen > self.max_coast_s:
                    continue
            survivors.append(s)
        self._tracks = survivors

        # 4. Detections with no track start a new, unconfirmed one.
        for di, d in enumerate(detections):
            if di not in matched_detections:
                self._tracks.append(_State(d.cx, d.cy, d.w, d.h, d.score, d.label, last_seen=t))

        return [self._export(s, coasting=s.last_seen < t) for s in self._tracks if s.id != 0]

    def _correct(self, s: _State, d: Detection, t: float) -> None:
        rx, ry = d.cx - s.cx, d.cy - s.cy  # how wrong the prediction was
        s.cx += self.alpha * rx
        s.cy += self.alpha * ry
        # The error built up since the last detection, which is longer than one
        # frame if the track was coasting. Dividing by the frame time instead
        # would turn every re-acquired target into a burst of false speed.
        elapsed = t - s.last_seen
        if elapsed > 0:
            s.vx += self.beta * rx / elapsed
            s.vy += self.beta * ry / elapsed
        s.w += self.alpha * (d.w - s.w)
        s.h += self.alpha * (d.h - s.h)
        s.score, s.label = d.score, d.label
        s.hits += 1
        s.last_seen = t
        if s.id == 0 and s.hits >= self.confirm_hits:
            s.id = self._next_id
            self._next_id += 1

    @staticmethod
    def _export(s: _State, coasting: bool) -> Track:
        return Track(
            id=s.id, cx=s.cx, cy=s.cy, w=s.w, h=s.h, vx=s.vx, vy=s.vy,
            score=s.score, label=s.label, hits=s.hits, coasting=coasting,
        )  # fmt: skip
