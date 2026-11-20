import numpy as np

from livedetect.render import VIEW_WIDTH, draw, draw_view
from livedetect.types import Frame, Result, Track, View


def result(view: View | None, coasting: bool = False) -> Result:
    image = np.zeros((540, 960, 3), np.uint8)
    image[250:260, 590:610] = (0, 0, 255)  # a red target at (600, 255)
    track = Track(7, 600.0, 255.0, 20.0, 10.0, 0.0, 0.0, 0.9, "drone", 5, coasting)
    return Result(Frame(0, 0.0, image), (track,), 1, 0.0, 0.0, view)


def test_the_frame_is_drawn_on_a_copy():
    r = result(View(600, 255, 240, 135, 4.0, 7))
    out = draw(r)
    assert out.shape == r.frame.image.shape
    assert out is not r.frame.image and (out != r.frame.image).any()
    assert r.frame.image[100, 100].tolist() == [0, 0, 0]  # the original is untouched


def test_the_view_shows_its_window_enlarged_with_the_target_in_the_middle():
    out = draw_view(result(View(cx=600, cy=255, w=240, h=135, zoom=4.0, target=7)))
    assert out.shape == (270, VIEW_WIDTH, 3)
    middle = out[135, 240].astype(int)
    assert middle[2] > 200 and middle[0] < 60  # the red target, now at the centre
    # The 20-pixel target is 4x larger than in the frame... and then twice more,
    # because the 240-pixel window is shown 480 pixels wide.
    red = (out[135, :, 2] > 200) & (out[135, :, 0] < 60)
    assert 34 <= red.sum() <= 46


def test_the_view_moves_by_fractions_of_a_pixel():
    a = draw_view(result(View(600.0, 255, 240, 135, 4.0, 7)))
    b = draw_view(result(View(600.25, 255, 240, 135, 4.0, 7)))
    assert (a != b).any()


def test_without_a_view_the_whole_frame_is_shown():
    out = draw_view(result(None))
    assert out.shape == (270, VIEW_WIDTH, 3)
