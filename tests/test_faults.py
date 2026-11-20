import time

import pytest

from helpers import FakeDetector, FakeSource, wait_until
from livedetect.controls import Controls
from livedetect.detectors import TiledDetector
from livedetect.faults import Faults, FaultyDetector, FaultySource
from livedetect.pipeline import QUEUE_FRAMES, Pipeline
from livedetect.sources import SourceError


def test_a_cut_source_fails_on_open_and_on_read():
    faults = Faults()
    source = FaultySource(FakeSource(), faults)
    source.open()
    assert source.read().shape == (8, 8, 3)
    faults.camera_cut = True
    with pytest.raises(SourceError):
        source.read()
    with pytest.raises(SourceError):
        source.open()


def test_a_slowed_detector_takes_longer_and_still_detects():
    faults = Faults(detector_delay_s=0.05)
    detector = FaultyDetector(FakeDetector(), faults)
    started = time.monotonic()
    assert len(detector.detect(None)) == 1
    assert time.monotonic() - started >= 0.05


@pytest.fixture
def rig():
    """A running pipeline with fault wrappers and its controls."""
    faults = Faults()
    tiled = TiledDetector(FakeDetector())
    pipeline = Pipeline(
        FaultySource(FakeSource(period_s=0.002), faults),
        FaultyDetector(tiled, faults),
        backoff_min_s=0.01,
        backoff_max_s=0.02,
    )
    pipeline.start()
    yield pipeline, Controls(pipeline, faults, tiled)
    pipeline.stop()


def test_cutting_the_camera_stops_the_stream_and_reconnecting_restores_it(rig):
    pipeline, controls = rig
    assert wait_until(lambda: pipeline.metrics.state == "running")
    controls.apply({"camera_cut": True})
    assert wait_until(lambda: pipeline.metrics.state == "reconnecting")
    assert wait_until(lambda: pipeline.metrics.snapshot()["source_errors"] >= 2)  # it keeps trying
    controls.apply({"camera_cut": False})
    assert wait_until(lambda: pipeline.metrics.state == "running")


def test_a_queue_makes_results_late_and_the_slot_does_not(rig):
    # Frames every 2 ms, 20 ms to process each. Behind a full queue of 30 a
    # frame waits 30 x 20 ms; behind the one-frame slot it waits at most 20 ms.
    pipeline, controls = rig
    controls.apply({"detector_delay_ms": 20})

    def latency() -> float:
        return pipeline.metrics.snapshot()["latency_now_ms"] or 0.0

    controls.apply({"queue": True})
    assert wait_until(lambda: pipeline.waiting == QUEUE_FRAMES)
    assert wait_until(lambda: latency() > 0.5 * QUEUE_FRAMES * 20)

    controls.apply({"queue": False})
    assert wait_until(lambda: 0 < latency() < 100)
    assert pipeline.waiting <= 1


def test_controls_report_their_state(rig):
    _, controls = rig
    controls.apply({"detector_delay_ms": 150, "tiles": "3x2"})
    assert controls.state() == {
        "detector_delay_ms": 150,
        "camera_cut": False,
        "queue": False,
        "tiles": "3x2",
    }
    assert controls.tiled.grid == (3, 2)


@pytest.mark.parametrize(
    "bad",
    [
        {"detector_delay_ms": -1},
        {"detector_delay_ms": "slow"},
        {"camera_cut": "yes"},
        {"queue": 1},
        {"tiles": "9x9"},
        {"tiles": 6},
        {"self_destruct": True},
    ],
)
def test_an_invalid_change_is_refused_and_nothing_else_is_applied(rig, bad):
    _, controls = rig
    with pytest.raises(ValueError):
        controls.apply({"camera_cut": True, **bad} if "camera_cut" not in bad else bad)
    assert controls.state()["camera_cut"] is False


def test_tiles_are_refused_for_a_detector_that_cannot_be_tiled():
    pipeline = Pipeline(FakeSource(), FakeDetector())
    controls = Controls(pipeline, Faults(), tiled=None)
    assert controls.state()["tiles"] is None
    with pytest.raises(ValueError):
        controls.apply({"tiles": "3x2"})
