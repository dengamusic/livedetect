import pytest

from livedetect.tracker import Tracker
from livedetect.types import Detection

DT = 1 / 30


def detection(x: float, y: float) -> Detection:
    return Detection(x, y, 4.0, 4.0, 0.9)


def test_track_is_reported_after_three_hits_and_keeps_its_id():
    tracker = Tracker(confirm_hits=3)
    ids = [[t.id for t in tracker.update([detection(100 + 4 * i, 50)], i * DT)] for i in range(10)]
    assert ids[0] == [] and ids[1] == []
    assert all(frame_ids == [1] for frame_ids in ids[2:])


def test_velocity_estimate_converges():
    tracker = Tracker()
    for i in range(40):  # 4 pixels per frame to the right at 30 fps = 120 px/s
        tracks = tracker.update([detection(100 + 4 * i, 50)], i * DT)
    assert tracks[0].vx == pytest.approx(120, rel=0.02)
    assert tracks[0].vy == pytest.approx(0, abs=1)


def test_one_frame_of_noise_is_never_reported():
    tracker = Tracker()
    for i in range(20):  # a false alarm somewhere new in every frame
        tracks = tracker.update([detection(50 + 97 * (i % 5), 40 + 61 * (i % 3))], i * DT)
        assert tracks == []


def test_track_coasts_through_missed_detections():
    tracker = Tracker(max_misses=10)
    for i in range(10):
        tracker.update([detection(100 + 4 * i, 50)], i * DT)
    for i in range(10, 15):  # detector misses five frames
        (track,) = tracker.update([], i * DT)
        assert track.coasting
        assert track.cx == pytest.approx(100 + 4 * i, abs=3)  # still moving at its velocity
    (track,) = tracker.update([detection(100 + 4 * 15, 50)], 15 * DT)
    assert track.id == 1 and not track.coasting


def test_track_is_dropped_after_max_misses():
    tracker = Tracker(max_misses=3)
    for i in range(5):
        tracker.update([detection(100, 50)], i * DT)
    seen = [len(tracker.update([], i * DT)) for i in range(5, 10)]
    assert seen == [1, 1, 1, 0, 0]


def test_crossing_targets_keep_their_ids():
    tracker = Tracker()
    for i in range(31):  # one flies right, one flies left; they pass 4 px apart
        right = detection(100 + 4 * i, 100)
        left = detection(220 - 4 * i, 104)
        tracks = tracker.update([right, left], i * DT)
        if i == 5:
            start = {t.id: t.vx > 0 for t in tracks}
    end = {t.id: t.vx > 0 for t in tracks}
    assert len(end) == 2
    assert end == start


def test_long_gap_discards_stale_tracks():
    tracker = Tracker(max_gap_s=1.0)
    for i in range(5):
        tracker.update([detection(100, 50)], i * DT)
    assert tracker.update([detection(100, 50)], 60.0) == []  # after a reconnect, start again
