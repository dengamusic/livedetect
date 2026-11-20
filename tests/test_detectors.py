import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from livedetect.detectors import (
    MotionDetector,
    OnnxYoloDetector,
    TiledDetector,
    TorchvisionDetector,
    decode_yolo,
    letterbox,
    make_detector,
    merge_detections,
    parse_grid,
    tile_boxes,
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


# --- tiling -------------------------------------------------------------------


class DotDetector:
    """Reports every bright blob it is shown, as a stand-in for a network.

    Blobs under `min_size` pixels in the image it is given are missed, which is
    how a real network behaves on targets that are too small for it.
    """

    def __init__(self, min_size: int = 1) -> None:
        self.min_size = min_size
        self.calls = 0

    def detect(self, image: np.ndarray) -> list[Detection]:
        self.calls += 1
        mask = (image[:, :, 0] > 128).astype(np.uint8)
        count, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
        return [
            Detection(float(centroids[i][0]), float(centroids[i][1]), float(w), float(h), 0.9)
            for i in range(1, count)
            for x, y, w, h, _ in [stats[i]]
            if max(w, h) >= self.min_size
        ]


def frame_with_dots(*dots: tuple[int, int, int]) -> np.ndarray:
    image = np.zeros((540, 960, 3), np.uint8)
    for x, y, size in dots:
        image[y : y + size, x : x + size] = 255
    return image


@pytest.mark.parametrize("grid", [(2, 2), (3, 2), (5, 3)])
def test_tiles_cover_the_frame_and_overlap_their_neighbours(grid):
    cols, rows = grid
    tiles = tile_boxes(960, 540, cols, rows, overlap=0.2)
    assert len(tiles) == cols * rows
    covered = np.zeros((540, 960), np.uint8)
    for x0, y0, x1, y1 in tiles:
        assert 0 <= x0 < x1 <= 960 and 0 <= y0 < y1 <= 540
        covered[y0:y1, x0:x1] += 1
    assert covered.min() >= 1  # no gap
    assert covered.max() >= 2  # neighbours share a strip
    first, second = tiles[0], tiles[1]
    shared = first[2] - second[0]
    assert shared == pytest.approx(0.2 * (first[2] - first[0]), abs=2)


def test_one_by_one_grid_is_the_plain_detector():
    inner = DotDetector()
    found = TiledDetector(inner).detect(frame_with_dots((100, 100, 6)))
    assert inner.calls == 1 and len(found) == 1


def test_tiling_reports_positions_in_frame_coordinates():
    image = frame_with_dots((700, 400, 6))  # in the bottom-right tiles
    inner = DotDetector()
    (found,) = TiledDetector(inner, 3, 2).detect(image)
    assert inner.calls == 3 * 2 + 1  # every tile, plus the whole frame
    assert (found.cx, found.cy) == pytest.approx((702.5, 402.5))


def test_a_target_seen_in_two_overlapping_tiles_is_reported_once():
    tiles = tile_boxes(960, 540, 3, 2)
    seam_x = (tiles[1][0] + tiles[0][2]) // 2  # the middle of the strip tiles 0 and 1 share
    found = TiledDetector(DotDetector(), 3, 2).detect(frame_with_dots((seam_x, 100, 6)))
    assert len(found) == 1


def test_a_target_cut_by_a_tile_edge_is_not_reported_as_a_fragment():
    tiles = tile_boxes(960, 540, 3, 2)
    edge = tiles[0][2]  # right edge of the first tile
    image = frame_with_dots((edge - 3, 100, 10))  # straddles it
    found = TiledDetector(DotDetector(), 3, 2).detect(image)
    assert [(d.w, d.h) for d in found] == [(10.0, 10.0)]  # whole, once


def test_merge_keeps_the_best_box_per_target_and_separates_neighbours():
    sloppy = Detection(100, 100, 30, 30, 0.6)  # the same drone, seen small in the whole frame
    tight = Detection(103, 101, 10, 8, 0.9)  # and large in a tile
    other = Detection(160, 100, 10, 8, 0.7)
    assert merge_detections([sloppy, tight, other]) == [tight, other]


def test_grid_text_is_parsed_and_checked():
    assert parse_grid("3x2") == (3, 2)
    assert parse_grid("5X3") == (5, 3)
    for bad in ("3", "axb", "0x2", "9x9", "3x2x1"):
        with pytest.raises(ValueError):
            parse_grid(bad)


def test_motion_detector_refuses_tiles():
    assert isinstance(make_detector("motion"), MotionDetector)
    with pytest.raises(ValueError):
        make_detector("motion", tiles=(3, 2))


def test_tiling_finds_a_drone_too_small_for_the_whole_frame_pass():
    path = cache_dir() / MODELS["drone"].filename
    if not path.exists():
        pytest.skip("drone model not downloaded; run `python -m livedetect.models`")
    clip = cv2.VideoCapture(str(Path(__file__).parent.parent / "assets" / "drone_tiny.mp4"))
    clip.set(cv2.CAP_PROP_POS_FRAMES, 60)  # two seconds in: the drone is 7 pixels wide here
    network = OnnxYoloDetector(path)
    plain = tiled = frames = 0
    for _ in range(30):
        ok, image = clip.read()
        assert ok
        frames += 1
        plain += bool(network.detect(image))
        tiled += bool(TiledDetector(network, 3, 2).detect(image))
    assert plain <= 0.3 * frames  # measured: 1 of 30
    assert tiled >= 0.8 * frames  # measured: 29 of 30
