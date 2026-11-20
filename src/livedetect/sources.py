"""Where frames come from.

Every source has the same three methods, so the pipeline does not care whether
it is reading a network camera, a file or the built-in synthetic scene:

    open()   connect; raise SourceError if that fails
    read()   block until the next frame (BGR uint8); raise SourceError when the
             stream breaks and EndOfStream when it ends
    close()  release everything; safe to call twice or without open()
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np


class SourceError(RuntimeError):
    """The source cannot deliver frames right now."""


class EndOfStream(SourceError):
    """The stream ended cleanly (end of a file, or the far side hung up)."""


class FrameSource(Protocol):
    def open(self) -> None: ...
    def read(self) -> np.ndarray: ...
    def close(self) -> None: ...


class SyntheticSource:
    """A sky with a few small dark objects flying across it.

    Needs no camera and no files, and the true object positions are known
    (`target_positions`), which is what the tests check the detector against.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 360,
        fps: float = 30.0,
        n_targets: int = 3,
        seed: int = 0,
        noise: float = 2.0,
        radius: int = 3,
        pace: bool = True,
    ) -> None:
        self.width, self.height, self.fps = width, height, fps
        self.noise, self.radius, self.pace, self.seed = noise, radius, pace, seed
        rng = np.random.default_rng(seed)
        margin = 20
        self._lo = np.array([margin, margin], float)
        self._span = np.array([width - 2 * margin, height - 2 * margin], float)
        self._start = self._lo + rng.random((n_targets, 2)) * self._span
        speed = rng.uniform(60, 150, n_targets) / fps  # pixels per frame
        heading = rng.uniform(0, 2 * np.pi, n_targets)
        self._step = np.stack([speed * np.cos(heading), speed * np.sin(heading)], axis=1)
        # Sky: light blue at the top fading to near white at the horizon.
        t = np.linspace(0.0, 1.0, height)[:, None, None]
        top, bottom = np.array([235, 190, 140.0]), np.array([240, 230, 220.0])
        self._sky = np.broadcast_to(top + (bottom - top) * t, (height, width, 3)).copy()
        self._index = 0
        self._t0 = 0.0

    def target_positions(self, index: int) -> np.ndarray:
        """True (x, y) of every object in frame `index`; they bounce off the edges."""
        m = (self._start - self._lo + self._step * index) % (2 * self._span)
        return self._lo + self._span - np.abs(m - self._span)

    def render(self, index: int) -> np.ndarray:
        rng = np.random.default_rng((self.seed, index))
        image = self._sky + rng.normal(0.0, self.noise, (self.height, self.width, 1))
        image = np.clip(image, 0, 255).astype(np.uint8)
        for x, y in self.target_positions(index):
            centre = (int(round(x)), int(round(y)))
            cv2.circle(image, centre, self.radius, (40, 40, 40), -1, cv2.LINE_AA)
        return image

    def open(self) -> None:
        self._index = 0
        self._t0 = time.monotonic()

    def read(self) -> np.ndarray:
        if self.pace:  # deliver frames on a fixed schedule, like a camera
            delay = self._t0 + self._index / self.fps - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        image = self.render(self._index)
        self._index += 1
        return image

    def close(self) -> None:
        pass


class OpenCVSource:
    """A video file, webcam index or URL read through OpenCV.

    A fallback for machines without GStreamer. Files are played at their own
    frame rate so they behave like a camera.
    """

    def __init__(self, target: str, pace: bool = True) -> None:
        self.target, self.pace = target, pace
        self._cap: cv2.VideoCapture | None = None
        self._period = 0.0
        self._next = 0.0

    def open(self) -> None:
        self._cap = cv2.VideoCapture(int(self.target) if self.target.isdigit() else self.target)
        if not self._cap.isOpened():
            raise SourceError(f"cannot open {self.target!r}")
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        is_file = os.path.exists(self.target)
        self._period = 1.0 / fps if (self.pace and is_file and fps > 0) else 0.0
        self._next = time.monotonic()

    def read(self) -> np.ndarray:
        if self._cap is None:
            raise SourceError("source is not open")
        ok, image = self._cap.read()
        if not ok:
            raise EndOfStream(f"no more frames from {self.target!r}")
        if self._period:
            delay = self._next - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self._next = max(self._next + self._period, time.monotonic())
        return image

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class GStreamerSource:
    """Frames from a GStreamer pipeline.

    Give it either a URI (``rtsp://...``, ``file:///...``, or a plain path) or
    the front half of a pipeline that ends in raw or decodable video, e.g.
    ``rtspsrc location=rtsp://cam/stream latency=100 ! decodebin``.

    The appsink at the end keeps one buffer and drops the rest, so a stalled
    reader gets the newest frame when it comes back instead of a backlog.
    """

    _TAIL = "videoconvert ! video/x-raw,format=BGR ! appsink name=sink max-buffers=1 drop=true"

    def __init__(
        self,
        uri: str | None = None,
        pipeline: str | None = None,
        timeout_s: float = 3.0,
    ) -> None:
        if (uri is None) == (pipeline is None):
            raise ValueError("give exactly one of uri or pipeline")
        if uri is not None:
            if "://" not in uri:
                uri = Path(uri).resolve().as_uri()
            # Play files in real time (sync=true); show live streams as they arrive.
            sync = "true" if uri.startswith("file://") else "false"
            self.description = f'uridecodebin uri="{uri}" ! {self._TAIL} sync={sync}'
        else:
            self.description = f"{pipeline} ! {self._TAIL} sync=false"
        self.timeout_s = timeout_s
        self._gst = None
        self._pipeline = None
        self._sink = None

    def open(self) -> None:
        try:
            import gi

            gi.require_version("Gst", "1.0")
            from gi.repository import GLib, Gst
        except (ImportError, ValueError) as exc:
            raise SourceError(f"GStreamer Python bindings are not available: {exc}") from exc
        Gst.init(None)
        self._gst = Gst
        try:
            self._pipeline = Gst.parse_launch(self.description)
        except GLib.Error as exc:
            raise SourceError(f"bad pipeline: {exc.message}") from exc
        self._sink = self._pipeline.get_by_name("sink")
        if self._pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            reason = self._bus_error() or "pipeline refused to start"
            raise SourceError(reason)

    def read(self) -> np.ndarray:
        if self._sink is None:
            raise SourceError("source is not open")
        Gst = self._gst
        sample = self._sink.emit("try-pull-sample", int(self.timeout_s * Gst.SECOND))
        if sample is None:
            error = self._bus_error()
            if error:
                raise SourceError(error)
            if self._sink.get_property("eos"):
                raise EndOfStream("stream ended")
            raise SourceError(f"no frame for {self.timeout_s:.1f} s")
        structure = sample.get_caps().get_structure(0)
        width, height = structure.get_value("width"), structure.get_value("height")
        buffer = sample.get_buffer()
        ok, info = buffer.map(Gst.MapFlags.READ)
        if not ok:
            raise SourceError("could not map video buffer")
        try:
            # GStreamer pads each row to a multiple of 4 bytes, so the row
            # length (stride) can be longer than width * 3.
            rows = np.frombuffer(info.data, np.uint8).reshape(height, -1)
            return rows[:, : width * 3].reshape(height, width, 3).copy()
        finally:
            buffer.unmap(info)

    def close(self) -> None:
        if self._pipeline is not None:
            self._pipeline.set_state(self._gst.State.NULL)
        self._pipeline = self._sink = None

    def _bus_error(self) -> str | None:
        Gst = self._gst
        message = self._pipeline.get_bus().pop_filtered(Gst.MessageType.ERROR)
        if message is None:
            return None
        error, _debug = message.parse_error()
        return error.message


def make_source(spec: str, width: int = 640, height: int = 360, fps: float = 30.0) -> FrameSource:
    """Turn a source string from the command line into a source object.

    synthetic          the built-in scene
    cv:<target>        OpenCV: a file path, URL or webcam index
    gst:<pipeline>     the front half of a GStreamer pipeline
    anything else      a URI or file path, read through GStreamer
    """
    if spec == "synthetic":
        return SyntheticSource(width=width, height=height, fps=fps)
    if spec.startswith("cv:"):
        return OpenCVSource(spec[3:])
    if spec.startswith("gst:"):
        return GStreamerSource(pipeline=spec[4:])
    return GStreamerSource(uri=spec)
