import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from livedetect.detectors import (
    MotionDetector,
    OnnxYoloDetector,
    TorchvisionDetector,
    decode_yolo,
    letterbox,
    make_detector,
)
from livedetect.models import MODELS, cache_dir
from livedetect.sources import SyntheticSource
from livedetect.types import Detection


def test_motion_detector_finds_every_target_and_nothing_else():
    source = SyntheticSource(n_targets=3, seed=1, pace=False)
    detector = MotionDetector()
    for i in range(120):
        detections = detector.detect(source.render(i))
        if i < 45:  # background still settling
            continue
        truth = source.target_positions(i)
        for x, y in truth:  # every target is found, within a few pixels
            assert min(math.hypot(d.cx - x, d.cy - y) for d in detections) < 4, f"frame {i}"
        for d in detections:  # and nothing is reported away from a target
            assert min(math.hypot(d.cx - x, d.cy - y) for x, y in truth) < 10, f"frame {i}"


def test_motion_detector_is_quiet_on_a_static_scene():
    source = SyntheticSource(n_targets=0, pace=False)
    detector = MotionDetector()
    assert all(detector.detect(source.render(i)) == [] for i in range(80))


def test_motion_detector_reports_nothing_during_warmup():
    source = SyntheticSource(n_targets=3, pace=False)
    detector = MotionDetector(warmup_frames=30)
    assert all(detector.detect(source.render(i)) == [] for i in range(30))


def test_motion_detector_starts_over_when_the_frame_size_changes():
    detector = MotionDetector(warmup_frames=2)
    for _ in range(5):
        detector.detect(np.zeros((48, 64, 3), np.uint8))
    assert detector.detect(np.full((96, 128, 3), 255, np.uint8)) == []


def test_make_detector_rejects_unknown_names():
    with pytest.raises(ValueError, match="unknown detector"):
        make_detector("nope")


def test_torchvision_detector_returns_well_formed_detections():
    pytest.importorskip("torchvision")
    detector = TorchvisionDetector(pretrained=False, score_threshold=0.0, device="cpu")
    detections = detector.detect(np.zeros((240, 320, 3), np.uint8))
    assert all(isinstance(d, Detection) for d in detections)
    assert all(0.0 <= d.score <= 1.0 and d.w >= 0 and d.h >= 0 for d in detections)


# --- ONNX / YOLO -------------------------------------------------------------


def test_letterbox_pads_to_a_square_and_reports_the_mapping():
    image = np.zeros((360, 640, 3), np.uint8)
    canvas, scale, left, top = letterbox(image, 416)
    assert canvas.shape == (416, 416, 3)
    assert scale == pytest.approx(0.65) and (left, top) == (0, 91)
    assert (canvas[0] == 114).all()  # padding above the image
    assert (canvas[208] == 0).all()  # the image itself


def test_decode_yolo_maps_boxes_back_and_keeps_one_box_per_object():
    # A 640x360 image letterboxed into 416: scale 0.65, 91 pixels of padding on top.
    rows = np.array(
        [
            # cx,  cy,   w,  h, objectness, class score
            [208, 208, 26, 13, 0.9, 0.9],  # a drone in the middle of the image
            [209, 208, 26, 13, 0.8, 0.9],  # the same drone again, less confident
            [65, 130, 13, 13, 0.9, 1.0],  # a second drone
            [300, 300, 10, 10, 0.2, 0.5],  # too weak to report
        ],
        np.float32,
    )
    detections = decode_yolo(rows, 0.65, 0, 91, 0.4, 0.45, {0: "drone"})
    detections.sort(key=lambda d: d.cx)
    assert [d.label for d in detections] == ["drone", "drone"]
    second, middle = detections
    assert (second.cx, second.cy) == pytest.approx((100, 60))
    assert (middle.cx, middle.cy, middle.w, middle.h) == pytest.approx((320, 180, 40, 20))
    assert middle.score == pytest.approx(0.81)


def test_decode_yolo_returns_nothing_when_nothing_is_confident():
    rows = np.array([[100, 100, 10, 10, 0.1, 0.9]], np.float32)
    assert decode_yolo(rows, 1.0, 0, 0, 0.4, 0.45, {0: "drone"}) == []


def test_drone_model_finds_the_drone_in_the_bundled_clip():
    """Runs only where the model is already downloaded: `python -m livedetect.models`."""
    pytest.importorskip("onnxruntime")
    model = cache_dir() / MODELS["drone"].filename
    clip = Path(__file__).parent.parent / "assets" / "drone_close.mp4"
    if not model.exists():
        pytest.skip("drone model is not in the cache")
    detector = OnnxYoloDetector(model)
    video = cv2.VideoCapture(str(clip))
    hits = []
    for index in range(600):  # the first 20 seconds; the drone is in view throughout
        ok, frame = video.read()
        assert ok
        if index % 15 == 0:
            detections = detector.detect(frame)
            hits.append(len(detections) == 1 and detections[0].label == "drone")
    assert sum(hits) / len(hits) >= 0.9
