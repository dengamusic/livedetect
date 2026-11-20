import json
import time

import numpy as np
import pytest

from helpers import FakeDetector, FakeSource
from livedetect.pipeline import Pipeline
from livedetect.server import create_app
from livedetect.types import Frame, Result, Track


@pytest.fixture
def pipeline() -> Pipeline:
    # Not started: the tests publish results by hand, so no threads or timing.
    return Pipeline(FakeSource(), FakeDetector())


@pytest.fixture
def client(pipeline):
    return create_app(pipeline).test_client()


def publish(pipeline: Pipeline, seq: int = 0) -> Result:
    frame = Frame(seq, time.monotonic(), np.zeros((48, 64, 3), np.uint8))
    track = Track(
        id=1, cx=30.0, cy=20.0, w=4.0, h=4.0, vx=120.0, vy=0.0,
        score=0.9, label="moving", hits=5, coasting=False,
    )  # fmt: skip
    result = Result(frame, (track,), n_detections=1, processing_s=0.004, latency_s=0.006)
    pipeline.results.publish(result)
    return result


def test_index_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"/video.mjpg" in response.data


def test_health_is_503_before_any_frame(client):
    response = client.get("/healthz")
    assert response.status_code == 503
    assert response.json["status"] == "degraded"


def test_health_is_200_while_frames_flow(client, pipeline):
    pipeline.metrics.set_state("running")
    pipeline.metrics.frame_captured()
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json["status"] == "ok"


def test_health_is_503_while_reconnecting(client, pipeline):
    pipeline.metrics.frame_captured()
    pipeline.metrics.set_state("reconnecting")
    assert client.get("/healthz").status_code == 503


def test_metrics_reports_counters_and_latency(client, pipeline):
    pipeline.metrics.frame_captured()
    pipeline.metrics.frame_processed(latency_s=0.010, processing_s=0.004)
    pipeline.metrics.count("dropped", 2)
    body = client.get("/metrics").json
    assert body["captured"] == 1 and body["processed"] == 1 and body["dropped"] == 2
    assert body["latency_ms"]["p50"] == pytest.approx(10.0)


def test_tracks_is_404_until_there_is_a_result(client, pipeline):
    assert client.get("/tracks").status_code == 404
    publish(pipeline, seq=7)
    body = client.get("/tracks").json
    assert body["seq"] == 7
    assert body["tracks"][0]["id"] == 1 and body["tracks"][0]["vx"] == 120.0


def test_events_stream_sends_results_as_json(client, pipeline):
    publish(pipeline, seq=3)
    response = client.get("/events")
    assert response.mimetype == "text/event-stream"
    chunk = next(iter(response.response)).decode()
    response.close()
    assert chunk.startswith("data: ") and chunk.endswith("\n\n")
    assert json.loads(chunk[len("data: ") :])["seq"] == 3


def test_video_stream_sends_jpeg_frames(client, pipeline):
    publish(pipeline)
    response = client.get("/video.mjpg")
    assert response.mimetype == "multipart/x-mixed-replace"
    chunk = next(iter(response.response))
    response.close()
    assert chunk.startswith(b"--frame\r\nContent-Type: image/jpeg")
    assert b"\xff\xd8" in chunk  # JPEG start-of-image marker


def test_stream_unsubscribes_when_the_client_leaves(client, pipeline):
    publish(pipeline)
    response = client.get("/events")
    next(iter(response.response))
    assert pipeline.results.subscriber_count == 1
    response.close()
    assert pipeline.results.subscriber_count == 0
