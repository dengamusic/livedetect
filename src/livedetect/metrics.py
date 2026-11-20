"""Counters and timings for the pipeline, safe to update from any thread."""

from __future__ import annotations

import threading
import time
from collections import Counter, deque
from collections.abc import Callable

import numpy as np


class Metrics:
    def __init__(
        self,
        window: int = 300,
        rate_window_s: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._started = clock()
        self._state = "connecting"
        self._counts: Counter[str] = Counter()
        self._latency = deque(maxlen=window)  # seconds, last `window` results
        self._processing = deque(maxlen=window)
        self._rate_window_s = rate_window_s
        self._captured_at: deque[float] = deque()
        self._processed_at: deque[float] = deque()

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

    def frame_processed(self, latency_s: float, processing_s: float) -> None:
        with self._lock:
            self._counts["processed"] += 1
            self._stamp(self._processed_at)
            self._latency.append(latency_s)
            self._processing.append(processing_s)

    def _stamp(self, stamps: deque[float]) -> None:
        now = self._clock()
        stamps.append(now)
        self._forget_old(stamps, now)  # keeps memory bounded even if nobody reads

    def _forget_old(self, stamps: deque[float], now: float) -> None:
        while stamps and stamps[0] < now - self._rate_window_s:
            stamps.popleft()

    def _rate(self, stamps: deque[float], now: float) -> float:
        """Events per second over the last `rate_window_s`; 0 when nothing arrives."""
        self._forget_old(stamps, now)
        return len(stamps) / self._rate_window_s

    def snapshot(self) -> dict:
        with self._lock:
            now = self._clock()
            last = self._captured_at[-1] if self._captured_at else None
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
                "last_frame_age_s": None if last is None else round(now - last, 3),
                "latency_ms": _summary_ms(self._latency),
                "processing_ms": _summary_ms(self._processing),
            }


def _summary_ms(samples: deque[float]) -> dict[str, float] | None:
    if not samples:
        return None
    p50, p95 = np.percentile(np.fromiter(samples, float), [50, 95]) * 1e3
    return {"p50": round(p50, 2), "p95": round(p95, 2), "max": round(max(samples) * 1e3, 2)}
