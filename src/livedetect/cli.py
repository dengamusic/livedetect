"""Command line entry point: `python -m livedetect`.

Every option can also be set through an environment variable
(LIVEDETECT_SOURCE, LIVEDETECT_DETECTOR, ...), which is how the container is
configured.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import threading

from werkzeug.serving import make_server

from livedetect.detectors import make_detector
from livedetect.pipeline import Pipeline
from livedetect.server import create_app
from livedetect.sources import make_source

log = logging.getLogger("livedetect")


def _env(name: str, default: str) -> str:
    return os.environ.get(f"LIVEDETECT_{name}", default)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="livedetect", description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source",
        default=_env("SOURCE", "synthetic"),
        help="synthetic | rtsp://... | /path/to/clip.mp4 | gst:<pipeline> | cv:<path or index>",
    )
    parser.add_argument(
        "--detector",
        default=_env("DETECTOR", "motion"),
        choices=["motion", "drone", "ssdlite", "fasterrcnn"],
        help="motion needs no model; drone downloads an ONNX model (onnx extra); "
        "ssdlite and fasterrcnn are general-purpose PyTorch models (torch extra)",
    )
    parser.add_argument("--score-threshold", type=float, default=float(_env("SCORE", "0.4")))
    parser.add_argument("--host", default=_env("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(_env("PORT", "8000")))
    parser.add_argument("--log-level", default=_env("LOG_LEVEL", "INFO"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    # The page polls /metrics twice a second; one log line per request buries everything else.
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    pipeline = Pipeline(
        source=make_source(args.source),
        detector=make_detector(args.detector, args.score_threshold),
    )
    server = make_server(args.host, args.port, create_app(pipeline), threaded=True)

    # The server runs in its own thread so the main thread is free to wait for
    # Ctrl-C or `docker stop` and then shut both parts down in order.
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())

    pipeline.start()
    threading.Thread(target=server.serve_forever, name="http", daemon=True).start()
    log.info("serving on http://%s:%d  (source: %s)", args.host, args.port, args.source)
    stop.wait()

    log.info("shutting down")
    pipeline.stop()
    server.shutdown()
    return 0
