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
    misses: int = 0
    id: int = 0  # 0 until the track is confirmed


class Tracker:
    """Constant-velocity tracker with an alpha-beta filter.

    gate_px       how far a detection may be from a track's predicted position
    confirm_hits  detections needed before a track is reported (rejects noise)
    max_misses    frames a confirmed track survives without a detection
    alpha, beta   how strongly a detection corrects position and velocity
    max_gap_s     a pause longer than this (a reconnect) discards all tracks
    """

    def __init__(
        self,
        gate_px: float = 30.0,
        confirm_hits: int = 3,
        max_misses: int = 15,
        alpha: float = 0.6,
        beta: float = 0.3,
        max_gap_s: float = 1.0,
    ) -> None:
        self.gate_px, self.confirm_hits, self.max_misses = gate_px, confirm_hits, max_misses
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
            gate = self.gate_px + 0.5 * max(s.w, s.h)
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
            self._correct(self._tracks[ti], detections[di], dt)

        # 3. Tracks with no detection coast on their prediction, then die.
        survivors = []
        for ti, s in enumerate(self._tracks):
            if ti not in matched_tracks:
                s.misses += 1
                confirmed = s.id != 0
                if not confirmed or s.misses > self.max_misses:
                    continue
            survivors.append(s)
        self._tracks = survivors

        # 4. Detections with no track start a new, unconfirmed one.
        for di, d in enumerate(detections):
            if di not in matched_detections:
                self._tracks.append(_State(d.cx, d.cy, d.w, d.h, d.score, d.label))

        return [self._export(s) for s in self._tracks if s.id != 0]

    def _correct(self, s: _State, d: Detection, dt: float) -> None:
        rx, ry = d.cx - s.cx, d.cy - s.cy  # how wrong the prediction was
        s.cx += self.alpha * rx
        s.cy += self.alpha * ry
        if dt > 0:
            s.vx += self.beta * rx / dt
            s.vy += self.beta * ry / dt
        s.w += self.alpha * (d.w - s.w)
        s.h += self.alpha * (d.h - s.h)
        s.score, s.label = d.score, d.label
        s.hits += 1
        s.misses = 0
        if s.id == 0 and s.hits >= self.confirm_hits:
            s.id = self._next_id
            self._next_id += 1

    @staticmethod
    def _export(s: _State) -> Track:
        return Track(
            id=s.id, cx=s.cx, cy=s.cy, w=s.w, h=s.h, vx=s.vx, vy=s.vy,
            score=s.score, label=s.label, hits=s.hits, coasting=s.misses > 0,
        )  # fmt: skip
