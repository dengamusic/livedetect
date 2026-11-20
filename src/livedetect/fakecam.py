"""A stand-in for a network camera: `python -m livedetect.fakecam`.

Reads frames from any source (the synthetic scene by default, or a clip),
encodes them as H.264 and serves them over TCP. Point the service at it with

    --source "gst:tcpclientsrc host=127.0.0.1 port=5000 ! tsdemux ! h264parse ! avdec_h264"

Stop this process and the service loses its camera; start it again and the
service reconnects. Needs GStreamer with the x264 encoder.
"""

from __future__ import annotations

import argparse
import logging
import signal

import numpy as np

from livedetect.sources import EndOfStream, make_source

log = logging.getLogger("livedetect.fakecam")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="livedetect-fakecam", description=__doc__.splitlines()[0])
    parser.add_argument("--source", default="synthetic", help="synthetic, or cv:/path/to/clip.mp4")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args(argv)
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Treat `docker stop` like Ctrl-C. As process 1 in a container, a signal
    # with no handler is ignored and Docker would wait 10 s before killing us.
    signal.signal(signal.SIGTERM, signal.default_int_handler)

    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    Gst.init(None)

    source = make_source(args.source, fps=args.fps)
    source.open()
    first = source.read()
    # The encoder wants even dimensions, and rows whose byte length is a
    # multiple of 4 need no padding, so trim a few pixels if necessary.
    height, width = first.shape[0] // 2 * 2, first.shape[1] // 4 * 4

    encoder = Gst.parse_launch(
        "appsrc name=src is-live=true format=time do-timestamp=true "
        f"caps=video/x-raw,format=BGR,width={width},height={height},framerate={args.fps}/1 "
        "! videoconvert "
        "! x264enc tune=zerolatency speed-preset=ultrafast key-int-max=15 "
        "! mpegtsmux "
        f"! tcpserversink host={args.host} port={args.port}"
    )
    appsrc = encoder.get_by_name("src")
    encoder.set_state(Gst.State.PLAYING)
    log.info("serving %s as %dx%d H.264 on tcp://%s:%d", args.source, width, height,
             args.host, args.port)  # fmt: skip

    image = first
    try:
        while True:
            data = np.ascontiguousarray(image[:height, :width]).tobytes()
            appsrc.emit("push-buffer", Gst.Buffer.new_wrapped(data))
            try:
                image = source.read()
            except EndOfStream:  # a clip: start it again
                source.close()
                source.open()
                image = source.read()
    except KeyboardInterrupt:
        pass
    finally:
        encoder.set_state(Gst.State.NULL)
        source.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
