"""A tracking camera: a window that keeps one track in the middle of its view.

A real tracking camera sits on a motorised mount with a zoom lens. Here the
mount is a crop window that moves over the frame, and the lens is the size of
that window. The control law is the one a mount needs:

    pan rate = error / time constant + target velocity,  limited to a top speed

The first term closes the distance to the target. The second keeps up with a
target that is moving, so the window does not trail behind it. The limit is
what makes the motion look like a camera and not like a jump cut.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Sequence

from livedetect.types import Track, View


class Follower:
    """Chooses a track and steers the view onto it.

    fill         share of the view's height the target should take up
    max_zoom     closest zoom; 6 shows a sixth of the frame's width
    pan_time_s   time constant of the pan; smaller is snappier
    zoom_time_s  time constant of the zoom; slow, so box jitter does not pump it
    max_pan      top pan speed, in frame widths per second
    hold_s       after losing the target, wait this long before zooming out
    """

    def __init__(
        self,
        fill: float = 0.25,
        max_zoom: float = 6.0,
        pan_time_s: float = 0.25,
        zoom_time_s: float = 0.8,
        max_pan: float = 1.5,
        hold_s: float = 1.0,
    ) -> None:
        self.fill, self.max_zoom = fill, max_zoom
        self.pan_time_s, self.zoom_time_s = pan_time_s, zoom_time_s
        self.max_pan, self.hold_s = max_pan, hold_s
        self._lock = threading.Lock()
        self._requested: tuple[float, float] | None = None  # a click, not yet acted on
        self._target: int | None = None
        self._lost_at: float | None = None
        self._last_t: float | None = None
        self._cx = self._cy = 0.0
        self._zoom = 1.0

    def lock_at(self, x: float, y: float) -> None:
        """Follow the track nearest to this point of the frame. Safe from any thread."""
        with self._lock:
            self._requested = (x, y)

    def update(self, tracks: Sequence[Track], width: int, height: int, t: float) -> View:
        """Advance to time `t` (seconds) and return where the view is now."""
        if self._last_t is None:
            self._cx, self._cy = width / 2, height / 2
        # A long gap (a reconnect) is not a reason for one enormous step.
        dt = 0.0 if self._last_t is None else min(max(t - self._last_t, 0.0), 0.5)
        self._last_t = t

        target = self._choose(tracks)
        if target is not None:
            self._lost_at = None
            aim_x, aim_y, aim_vx, aim_vy = target.cx, target.cy, target.vx, target.vy
            # Fit the target's height, or its width if it is wider than the view's shape.
            window_h = max(target.h, target.w * height / width) / self.fill
            aim_zoom = min(max(height / max(window_h, 1.0), 1.0), self.max_zoom)
        else:
            if self._lost_at is None:
                self._lost_at = t
            aim_vx = aim_vy = 0.0
            if t - self._lost_at < self.hold_s:  # it may come straight back
                aim_x, aim_y, aim_zoom = self._cx, self._cy, self._zoom
            else:  # give up and look at everything again
                aim_x, aim_y, aim_zoom = width / 2, height / 2, 1.0

        # 1 - exp(-dt / T) is the share of the error a first-order system closes
        # in dt. Unlike dt / T it cannot exceed 1, so a slow frame rate makes
        # the view sluggish but never makes it overshoot.
        gain = 1.0 - math.exp(-dt / self.pan_time_s)
        limit = self.max_pan * width * dt
        self._cx += _limited(_step(self._cx, aim_x, aim_vx, dt, gain), limit)
        self._cy += _limited(_step(self._cy, aim_y, aim_vy, dt, gain), limit)
        # Zoom is a ratio, so it is eased in logarithms: 1x -> 2x takes as long as 2x -> 4x.
        zoom_gain = 1.0 - math.exp(-dt / self.zoom_time_s)
        self._zoom *= (aim_zoom / self._zoom) ** zoom_gain

        # The window cannot leave the frame, as a mount cannot turn past its stops.
        w, h = width / self._zoom, height / self._zoom
        self._cx = min(max(self._cx, w / 2), width - w / 2)
        self._cy = min(max(self._cy, h / 2), height - h / 2)
        return View(self._cx, self._cy, w, h, self._zoom, None if target is None else target.id)

    def _choose(self, tracks: Sequence[Track]) -> Track | None:
        with self._lock:
            requested, self._requested = self._requested, None
        if requested is not None and tracks:
            x, y = requested
            self._target = min(tracks, key=lambda t: math.hypot(t.cx - x, t.cy - y)).id
        for track in tracks:
            if track.id == self._target:
                return track  # stay on it, also while it is coasting
        # No target, or it is gone: take the track seen most often, which is
        # the one least likely to be a false alarm.
        seen = [t for t in tracks if not t.coasting]
        if not seen:
            self._target = None
            return None
        best = max(seen, key=lambda t: t.hits)
        self._target = best.id
        return best


def _step(position: float, aim: float, aim_velocity: float, dt: float, gain: float) -> float:
    """How far to move this frame: go along with the target, then close part of the rest.

    The target moved `aim_velocity * dt` since the last frame and `aim` already
    includes that. Moving the same distance keeps the error where it was;
    `gain` of that error is then removed. Adding the full error correction and
    the full target motion instead would put the view one frame ahead.
    """
    along = aim_velocity * dt
    return along + (aim - (position + along)) * gain


def _limited(step: float, limit: float) -> float:
    return max(-limit, min(limit, step))
