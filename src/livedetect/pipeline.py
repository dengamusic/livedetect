"""The pipeline: two threads joined by a one-frame mailbox.

    capture thread:  source.read() -> newest-frame slot   (reconnects on failure)
    process thread:  slot -> detector -> tracker -> tracking camera -> results broadcast

The camera sets the pace of the first thread and the detector the pace of the
second. When the detector is the slower one, the slot overwrites frames it has
not got to yet, so every result is about the present.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

from livedetect.buffers import Broadcast, FifoQueue, LatestSlot
from livedetect.detectors import Detector
from livedetect.follow import Follower
from livedetect.metrics import Metrics
from livedetect.sources import EndOfStream, FrameSource
from livedetect.tracker import Tracker
from livedetect.types import Frame, Result

log = logging.getLogger(__name__)

QUEUE_FRAMES = 30  # one second of camera frames; see Pipeline.set_queueing


class Pipeline:
    def __init__(
        self,
        source: FrameSource,
        detector: Detector,
        tracker: Tracker | None = None,
        follower: Follower | None = None,
        *,
        backoff_min_s: float = 0.5,
        backoff_max_s: float = 8.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.source = source
        self.detector = detector
        self.tracker = tracker or Tracker()
        self.follower = follower or Follower()
        self.queueing = False
        self.backoff_min_s, self.backoff_max_s = backoff_min_s, backoff_max_s
        self.metrics = Metrics(clock=clock)
        self.results: Broadcast[Result] = Broadcast()
        self._clock = clock
        self._frames: LatestSlot[Frame] | FifoQueue[Frame] = LatestSlot()
        self._stop = threading.Event()
        self._seq = 0
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        self._threads = [
            threading.Thread(target=self._capture_loop, name="capture", daemon=True),
            threading.Thread(target=self._process_loop, name="process", daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop both threads. A pipeline cannot be restarted; build a new one."""
        self._stop.set()
        self._frames.close()
        self.results.close()
        for thread in self._threads:
            thread.join(timeout)
        self.metrics.set_state("stopped")

    def set_queueing(self, on: bool) -> None:
        """Join the two threads with a queue instead of the one-frame slot.

        This is the design the pipeline is built to avoid, kept behind a switch
        so the two can be compared on the same stream: with a detector that
        cannot keep up, the queue fills and every result is `QUEUE_FRAMES`
        processing times old.
        """
        if on == self.queueing:
            return
        old = self._frames
        self._frames = FifoQueue(QUEUE_FRAMES) if on else LatestSlot()
        self.queueing = on
        self.metrics.count("dropped", old.clear())  # what was still waiting is abandoned
        old.close()  # wakes the process thread, which then reads from the new one

    @property
    def waiting(self) -> int:
        """Frames captured and not yet picked up by the detector."""
        return self._frames.size

    # --- capture thread -----------------------------------------------------

    def _capture_loop(self) -> None:
        backoff = self.backoff_min_s
        while not self._stop.is_set():
            delivered = self._run_connection()
            if self._stop.is_set():
                break
            self.metrics.set_state("reconnecting")
            self.metrics.count("reconnects")
            if delivered:
                # It was working a moment ago: retry at once. This is also what
                # makes a video file loop without a pause.
                backoff = self.backoff_min_s
            else:
                self._stop.wait(backoff)  # sleeps, but wakes immediately on stop()
                backoff = min(backoff * 2, self.backoff_max_s)

    def _run_connection(self) -> int:
        """Open the source and read until it fails or stop() is called.

        Returns how many frames arrived. Never raises: a broken source must not
        take the service down.
        """
        delivered = 0
        try:
            self.source.open()
            while not self._stop.is_set():
                image = self.source.read()
                frame = Frame(self._seq, self._clock(), image)
                self._seq += 1
                delivered += 1
                if delivered == 1:
                    self.metrics.set_state("running")
                self.metrics.frame_captured()
                if self._frames.put(frame):
                    self.metrics.count("dropped")
        except EndOfStream as exc:
            log.info("source ended: %s", exc)
        except Exception as exc:
            log.warning("source failed: %s", exc)
            self.metrics.count("source_errors")
        finally:
            try:
                self.source.close()
            except Exception:
                log.exception("closing the source failed")
        return delivered

    # --- process thread -----------------------------------------------------

    def _process_loop(self) -> None:
        while not self._stop.is_set():
            frame = self._frames.get(timeout=0.5)
            if frame is None:
                continue
            started = self._clock()
            try:
                detections = self.detector.detect(frame.image)
            except Exception:  # one bad frame must not stop the stream
                log.exception("detector failed on frame %d", frame.seq)
                self.metrics.count("detector_errors")
                continue
            tracks = self.tracker.update(detections, frame.captured_at)
            height, width = frame.image.shape[:2]
            view = self.follower.update(tracks, width, height, frame.captured_at)
            now = self._clock()
            result = Result(
                frame=frame,
                tracks=tuple(tracks),
                n_detections=len(detections),
                processing_s=now - started,
                latency_s=now - frame.captured_at,
                view=view,
            )
            self.metrics.frame_processed(
                result.latency_s, result.processing_s, found=bool(detections)
            )
            self.results.publish(result)
