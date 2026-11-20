import pytest

from livedetect.follow import Follower
from livedetect.types import Track

W, H = 960, 540


def track(id=1, cx=480.0, cy=270.0, w=20.0, h=10.0, vx=0.0, vy=0.0, hits=10, coasting=False):
    return Track(id, cx, cy, w, h, vx, vy, 0.9, "drone", hits, coasting)


def settle(follower, tracks, seconds=5.0, fps=30, start=0.0):
    """Feed the same tracks for a while and return the final view."""
    view = None
    for i in range(int(seconds * fps) + 1):
        view = follower.update(tracks, W, H, start + i / fps)
    return view


def test_starts_on_the_whole_frame():
    view = Follower().update([], W, H, 0.0)
    assert (view.cx, view.cy, view.w, view.h, view.zoom) == (W / 2, H / 2, W, H, 1.0)
    assert view.target is None


def test_centres_on_the_target_and_zooms_until_it_fills_the_view():
    follower = Follower(fill=0.25, max_zoom=10)
    view = settle(follower, [track(cx=300, cy=200, w=20, h=20)])
    assert view.target == 1
    assert view.cx == pytest.approx(300, abs=0.5) and view.cy == pytest.approx(200, abs=0.5)
    assert 20 / view.h == pytest.approx(0.25, rel=0.02)  # the target takes a quarter of the height
    assert view.w / view.h == pytest.approx(W / H)  # the view keeps the frame's shape


def test_zoom_stops_at_its_limit():
    view = settle(Follower(max_zoom=6), [track(w=4, h=4)])
    assert view.zoom == pytest.approx(6, rel=0.01)


def test_the_view_never_leaves_the_frame():
    view = settle(Follower(), [track(cx=5, cy=535, w=10, h=10)])
    assert view.cx - view.w / 2 >= -1e-6 and view.cy + view.h / 2 <= H + 1e-6


def test_pan_speed_is_limited():
    follower = Follower(max_pan=1.0)
    follower.update([], W, H, 0.0)
    view = follower.update([track(cx=900)], W, H, 0.1)  # the target is 420 px to the right
    assert view.cx - W / 2 <= 1.0 * W * 0.1 + 1e-6  # but 0.1 s allows 96 px at most


def test_keeps_up_with_a_moving_target_without_trailing_behind():
    follower, fps = Follower(max_zoom=3), 30
    view = None
    for i in range(5 * fps):
        t = i / fps
        view = follower.update([track(cx=200 + 80 * t, vx=80)], W, H, t)
    assert view.cx == pytest.approx(200 + 80 * t, abs=1.0)


def test_a_slow_frame_rate_does_not_make_it_overshoot():
    follower = Follower()
    follower.update([], W, H, 0.0)
    xs = [follower.update([track(cx=600)], W, H, 0.4 * (i + 1)).cx for i in range(10)]
    assert all(a <= b <= 600 + 1e-6 for a, b in zip(xs, xs[1:], strict=False))


def test_stays_on_its_target_when_a_better_looking_one_appears():
    follower = Follower()
    follower.update([track(id=1, hits=5)], W, H, 0.0)
    view = follower.update([track(id=1, hits=6), track(id=2, cx=100, hits=50)], W, H, 0.1)
    assert view.target == 1


def test_picks_the_most_established_track_first():
    view = Follower().update([track(id=1, hits=3), track(id=2, hits=40)], W, H, 0.0)
    assert view.target == 2


def test_follows_a_coasting_target_but_does_not_start_on_one():
    follower = Follower()
    assert follower.update([track(id=1, coasting=True)], W, H, 0.0).target is None
    follower.update([track(id=2)], W, H, 0.1)
    assert follower.update([track(id=2, coasting=True)], W, H, 0.2).target == 2


def test_a_click_selects_the_nearest_track():
    follower = Follower()
    tracks = [track(id=1, cx=100, hits=50), track(id=2, cx=800, hits=5)]
    assert follower.update(tracks, W, H, 0.0).target == 1
    follower.lock_at(780, 260)
    assert follower.update(tracks, W, H, 0.1).target == 2


def test_holds_then_zooms_out_when_the_target_is_lost():
    follower = Follower(hold_s=1.0)
    locked = settle(follower, [track(cx=300, cy=200)], seconds=5)
    held = follower.update([], W, H, 5.5)
    assert held.target is None
    assert (held.cx, held.zoom) == pytest.approx((locked.cx, locked.zoom))  # waiting for it
    released = settle(follower, [], seconds=8, start=6.0)
    assert released.zoom == pytest.approx(1.0, abs=0.01)
