import time

import cv2
import numpy as np
import pytest

from livedetect.sources import (
    EndOfStream,
    GStreamerSource,
    OpenCVSource,
    SourceError,
    SyntheticSource,
    make_source,
)


def _gstreamer_available() -> bool:
    try:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst  # noqa: F401
    except (ImportError, ValueError):
        return False
    return True


needs_gstreamer = pytest.mark.skipif(
    not _gstreamer_available(), reason="GStreamer Python bindings are not installed"
)


# --- synthetic ---------------------------------------------------------------


def test_synthetic_frames_have_the_requested_shape():
    source = SyntheticSource(width=320, height=180, pace=False)
    source.open()
    frame = source.read()
    assert frame.shape == (180, 320, 3) and frame.dtype == np.uint8


def test_synthetic_is_reproducible():
    a, b = SyntheticSource(seed=7, pace=False), SyntheticSource(seed=7, pace=False)
    assert np.array_equal(a.render(12), b.render(12))


def test_synthetic_targets_stay_inside_the_frame():
    source = SyntheticSource(width=320, height=180, n_targets=5, pace=False)
    for index in range(0, 5000, 37):
        xy = source.target_positions(index)
        assert (xy >= 0).all() and (xy[:, 0] < 320).all() and (xy[:, 1] < 180).all()


def test_synthetic_delivers_frames_at_its_frame_rate():
    source = SyntheticSource(width=64, height=48, fps=50, pace=True)
    source.open()
    started = time.monotonic()
    for _ in range(11):  # 10 intervals of 20 ms
        source.read()
    assert time.monotonic() - started >= 0.18


# --- OpenCV ------------------------------------------------------------------


def test_opencv_source_reads_a_file_then_reports_the_end(tmp_path):
    path = str(tmp_path / "clip.avi")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    for _ in range(5):
        writer.write(np.full((48, 64, 3), 128, np.uint8))
    writer.release()

    source = OpenCVSource(path, pace=False)
    source.open()
    frames = [source.read() for _ in range(5)]
    assert all(f.shape == (48, 64, 3) for f in frames)
    with pytest.raises(EndOfStream):
        source.read()
    source.close()
    source.close()  # closing twice is allowed


def test_opencv_source_fails_to_open_a_missing_file(tmp_path):
    with pytest.raises(SourceError):
        OpenCVSource(str(tmp_path / "missing.mp4")).open()


# --- GStreamer ---------------------------------------------------------------


@needs_gstreamer
def test_gstreamer_reads_frames_then_reports_the_end():
    source = GStreamerSource(
        pipeline="videotestsrc is-live=true num-buffers=5 "
        "! video/x-raw,width=64,height=48,framerate=30/1"
    )
    source.open()
    frames = []
    with pytest.raises(EndOfStream):
        for _ in range(10):
            frames.append(source.read())
    source.close()
    assert 1 <= len(frames) <= 5
    assert all(f.shape == (48, 64, 3) and f.dtype == np.uint8 for f in frames)


@needs_gstreamer
def test_gstreamer_strips_row_padding():
    # 62 pixels x 3 bytes = 186, which GStreamer pads to 188 bytes per row.
    # Reading the padding as pixels would smear the colour channels.
    source = GStreamerSource(
        pipeline="videotestsrc is-live=true pattern=red num-buffers=3 "
        "! video/x-raw,width=62,height=48,framerate=30/1"
    )
    source.open()
    frame = source.read()
    source.close()
    assert frame.shape == (48, 62, 3)
    blue, green, red = frame[..., 0], frame[..., 1], frame[..., 2]
    assert (red > 200).all() and (green < 50).all() and (blue < 50).all()


@needs_gstreamer
def test_gstreamer_rejects_a_pipeline_that_does_not_parse():
    with pytest.raises(SourceError, match="bad pipeline"):
        GStreamerSource(pipeline="no_such_element_xyz").open()


@needs_gstreamer
def test_gstreamer_reports_a_missing_file_as_an_error_not_an_end(tmp_path):
    source = GStreamerSource(uri=str(tmp_path / "missing.mp4"), timeout_s=2.0)
    with pytest.raises(SourceError) as raised:
        source.open()
        source.read()
    source.close()
    assert not isinstance(raised.value, EndOfStream)


# --- command-line spec -------------------------------------------------------


def test_make_source_picks_the_right_class():
    assert isinstance(make_source("synthetic"), SyntheticSource)
    assert isinstance(make_source("cv:0"), OpenCVSource)
    assert isinstance(make_source("gst:videotestsrc"), GStreamerSource)
    assert isinstance(make_source("rtsp://camera/stream"), GStreamerSource)
    assert 'uri="rtsp://camera/stream"' in make_source("rtsp://camera/stream").description
    assert "sync=true" in make_source("/tmp/clip.mp4").description  # files play in real time
