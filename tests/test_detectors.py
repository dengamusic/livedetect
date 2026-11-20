import math

import numpy as np
import pytest

from livedetect.detectors import MotionDetector, TorchvisionDetector, make_detector
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
