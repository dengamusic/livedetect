"""Counters and timings for the pipeline, safe to update from any thread."""

from __future__ import annotations

import threading
import time
from collections import Counter, deque
from collections.abc import Callable, Iterable

import numpy as np


class Metrics:
    """
    window_s       latency and processing-time statistics cover this many seconds
    rate_window_s  frame rates and the detection rate cover this many seconds
    """

    def __init__(
        self,
        window_s: float = 10.0,
        rate_window_s: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._started = clock()
        self._state = "connecting"
        self._counts: Counter[str] = Counter()
        self._window_s, self._rate_window_s = window_s, rate_window_s
        # One entry per result: (when, latency, processing time), all in seconds.
        self._results: deque[tuple[float, float, float]] = deque()
        self._captured_at: deque[float] = deque()
        self._processed_at: deque[float] = deque()
        self._found_at: deque[float] = deque()  # processed frames with a detection

    def set_state(self, state: str) -> None:
        with self._lock:
            self._state = state

    @property
    def state(self) -> str:
        return self._state

    def count(self, name: str, n: int = 1) -> None:
        with self._lock:
            self._counts[name] += n

    def frame_captured(self) -> None:
        with self._lock:
            self._counts["captured"] += 1
            self._stamp(self._captured_at)

    def frame_processed(self, latency_s: float, processing_s: float, found: bool = True) -> None:
        with self._lock:
            self._counts["processed"] += 1
            now = self._stamp(self._processed_at)
            if found:
                self._stamp(self._found_at)
            self._results.append((now, latency_s, processing_s))
            while self._results[0][0] < now - self._window_s:
                self._results.popleft()

    def _stamp(self, stamps: deque[float]) -> float:
        now = self._clock()
        stamps.append(now)
        self._forget_old(stamps, now)  # keeps memory bounded even if nobody reads
        return now

    def _forget_old(self, stamps: deque[float], now: float) -> None:
        while stamps and stamps[0] < now - self._rate_window_s:
            stamps.popleft()

    def _rate(self, stamps: deque[float], now: float) -> float:
        """Events per second over the last `rate_window_s`; 0 when nothing arrives."""
        self._forget_old(stamps, now)
        return len(stamps) / self._rate_window_s

    def _detection_rate(self, now: float) -> float | None:
        """Share of the frames processed in the last `rate_window_s` that had a detection."""
        self._forget_old(self._processed_at, now)
        self._forget_old(self._found_at, now)
        if not self._processed_at:
            return None
        return round(len(self._found_at) / len(self._processed_at), 2)

    def snapshot(self) -> dict:
        with self._lock:
            now = self._clock()
            last = self._captured_at[-1] if self._captured_at else None
            recent = [r for r in self._results if r[0] >= now - self._window_s]
            return {
                "state": self._state,
                "uptime_s": round(now - self._started, 1),
                "captured": self._counts["captured"],
                "processed": self._counts["processed"],
                "dropped": self._counts["dropped"],
                "reconnects": self._counts["reconnects"],
                "source_errors": self._counts["source_errors"],
                "detector_errors": self._counts["detector_errors"],
                "fps_in": round(self._rate(self._captured_at, now), 1),
                "fps_out": round(self._rate(self._processed_at, now), 1),
                "detection_rate": self._detection_rate(now),
                "last_frame_age_s": None if last is None else round(now - last, 3),
                "latency_now_ms": round(recent[-1][1] * 1e3, 2) if recent else None,
                "latency_ms": _summary_ms(r[1] for r in recent),
                "processing_ms": _summary_ms(r[2] for r in recent),
            }


def _summary_ms(samples: Iterable[float]) -> dict[str, float] | None:
    values = np.fromiter(samples, float) * 1e3
    if len(values) == 0:
        return None
    p50, p95 = np.percentile(values, [50, 95])
    return {"p50": round(p50, 2), "p95": round(p95, 2), "max": round(values.max(), 2)}
