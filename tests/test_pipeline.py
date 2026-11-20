import time

from helpers import FakeDetector, FakeSource, wait_until
from livedetect.pipeline import Pipeline
from livedetect.sources import EndOfStream


def run(pipeline: Pipeline, until, timeout: float = 5.0) -> dict:
    """Start the pipeline, wait for a condition on its metrics, stop it."""
    pipeline.start()
    try:
        assert wait_until(lambda: until(pipeline.metrics.snapshot()), timeout), (
            f"condition not reached: {pipeline.metrics.snapshot()}"
        )
        return pipeline.metrics.snapshot()
    finally:
        pipeline.stop()


def test_every_frame_is_processed_when_the_detector_keeps_up():
    pipeline = Pipeline(FakeSource(period_s=0.01), FakeDetector(delay_s=0.0))
    snapshot = run(pipeline, lambda m: m["processed"] >= 20)
    assert snapshot["dropped"] == 0
    assert snapshot["state"] == "running"


def test_slow_detector_drops_frames_and_latency_stays_bounded():
    # Frames arrive every 2 ms, each takes 30 ms to process. A queue would be
    # about a second behind after 30 results; the newest-frame slot is not.
    pipeline = Pipeline(FakeSource(period_s=0.002), FakeDetector(delay_s=0.03))
    snapshot = run(pipeline, lambda m: m["processed"] >= 30)
    assert snapshot["dropped"] > snapshot["processed"]
    assert snapshot["latency_ms"]["max"] < 300
    assert snapshot["latency_ms"]["p50"] < 100


def test_results_are_published_to_subscribers():
    pipeline = Pipeline(FakeSource(), FakeDetector())
    subscriber = pipeline.results.subscribe()
    pipeline.start()
    try:
        result = subscriber.get(timeout=5)
    finally:
        pipeline.stop()
    assert result is not None
    assert result.n_detections == 1
    assert result.latency_s >= result.processing_s >= 0


def test_reconnects_when_the_stream_breaks():
    source = FakeSource(frames_per_connection=5)
    pipeline = Pipeline(source, FakeDetector(), backoff_min_s=0.01)
    snapshot = run(pipeline, lambda m: m["reconnects"] >= 3 and m["captured"] >= 15)
    assert snapshot["source_errors"] >= 3
    assert source.opens >= 4
    assert source.closes >= 3  # every broken connection was closed before the next


def test_recovers_once_the_camera_comes_back():
    source = FakeSource(failing_opens=3)
    pipeline = Pipeline(source, FakeDetector(), backoff_min_s=0.01)
    pipeline.start()
    try:
        assert wait_until(lambda: pipeline.metrics.state == "reconnecting")
        assert wait_until(lambda: pipeline.metrics.state == "running")
        assert wait_until(lambda: pipeline.metrics.snapshot()["processed"] > 0)
    finally:
        pipeline.stop()
    assert source.opens == 4


def test_retries_back_off_while_the_camera_is_down():
    source = FakeSource(failing_opens=10**6)
    pipeline = Pipeline(source, FakeDetector(), backoff_min_s=0.05, backoff_max_s=0.2)
    pipeline.start()
    time.sleep(0.4)
    pipeline.stop()
    # Attempts at 0, 0.05, 0.15, 0.35 s. Without backoff there would be thousands.
    assert 2 <= source.opens <= 5


def test_end_of_stream_restarts_the_source_without_counting_an_error():
    source = FakeSource(frames_per_connection=5, error=EndOfStream)
    pipeline = Pipeline(source, FakeDetector())
    snapshot = run(pipeline, lambda m: m["captured"] >= 15)
    assert snapshot["source_errors"] == 0
    assert source.opens >= 3


def test_detector_failure_skips_the_frame_and_carries_on():
    pipeline = Pipeline(FakeSource(period_s=0.005), FakeDetector(fail_every=3))
    snapshot = run(pipeline, lambda m: m["detector_errors"] >= 3 and m["processed"] >= 6)
    assert snapshot["state"] == "running"


def test_stop_is_prompt_and_releases_subscribers():
    pipeline = Pipeline(FakeSource(failing_opens=10**6), FakeDetector(), backoff_min_s=30)
    subscriber = pipeline.results.subscribe()
    pipeline.start()
    assert wait_until(lambda: pipeline.metrics.state == "reconnecting")
    started = time.monotonic()
    pipeline.stop()  # must not wait out the 30 s backoff
    assert time.monotonic() - started < 1.0
    assert subscriber.closed
    assert pipeline.metrics.state == "stopped"
