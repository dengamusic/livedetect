"""The whole chain with real processes: an H.264 camera over TCP that is killed
and restarted while the pipeline is reading from it. Takes several seconds and
needs GStreamer with the x264 encoder; skipped where that is missing.
"""

import socket
import subprocess
import sys

import pytest

from helpers import wait_until
from livedetect.detectors import MotionDetector
from livedetect.pipeline import Pipeline
from livedetect.sources import GStreamerSource


def _encoder_available() -> bool:
    try:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
    except (ImportError, ValueError):
        return False
    Gst.init(None)
    return all(Gst.ElementFactory.find(name) for name in ("x264enc", "mpegtsmux", "avdec_h264"))


pytestmark = pytest.mark.skipif(
    not _encoder_available(), reason="GStreamer with x264 and MPEG-TS is not installed"
)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_camera(port: int) -> subprocess.Popen:
    command = [sys.executable, "-m", "livedetect.fakecam", "--host", "127.0.0.1", "--port", port]
    return subprocess.Popen([str(part) for part in command], stderr=subprocess.DEVNULL)


def test_pipeline_survives_the_camera_being_killed_and_restarted():
    port = _free_port()
    source = GStreamerSource(
        pipeline=f"tcpclientsrc host=127.0.0.1 port={port} ! tsdemux ! h264parse ! avdec_h264"
    )
    pipeline = Pipeline(source, MotionDetector(), backoff_min_s=0.2, backoff_max_s=0.5)
    camera = _start_camera(port)
    pipeline.start()
    try:
        # Camera up: frames decode, and the three synthetic targets get tracked.
        def tracking() -> bool:
            result = pipeline.results.latest
            return result is not None and len(result.tracks) == 3

        assert wait_until(tracking, timeout=30), pipeline.metrics.snapshot()

        # Camera dies.
        camera.kill()
        camera.wait()
        assert wait_until(lambda: pipeline.metrics.state == "reconnecting", timeout=10)

        # Camera comes back: frames flow again without restarting the service.
        seen = pipeline.metrics.snapshot()["captured"]
        camera = _start_camera(port)
        assert wait_until(
            lambda: pipeline.metrics.snapshot()["captured"] > seen + 30, timeout=30
        ), pipeline.metrics.snapshot()
        assert pipeline.metrics.state == "running"
        assert pipeline.metrics.snapshot()["reconnects"] >= 1
    finally:
        pipeline.stop()
        camera.kill()
        camera.wait()
