"""HTTP interface.

GET  /             live page
GET  /healthz      200 while frames are flowing, 503 otherwise
GET  /metrics      counters, frame rates, latency percentiles (JSON)
GET  /tracks       the most recent result (JSON)
GET  /events       results as they happen (server-sent events)
GET  /video.mjpg   the whole frame with tracks drawn on it (MJPEG)
GET  /follow.mjpg  the tracking camera's view (MJPEG)
POST /follow       {"x": .., "y": ..}: follow the track nearest to this point
GET  /controls     the state of the fault and mode switches
POST /controls     change them; refused unless started with --controls
"""

from __future__ import annotations

import json
from collections.abc import Iterator

from flask import Flask, Response, jsonify, request

from livedetect.controls import Controls
from livedetect.pipeline import Pipeline
from livedetect.render import JpegCache, draw, draw_view

STALE_AFTER_S = 2.0  # no frame for this long means the service is not healthy


def create_app(pipeline: Pipeline, controls: Controls | None = None) -> Flask:
    app = Flask(__name__, static_folder="static", static_url_path="/static")
    frames, views = JpegCache(draw), JpegCache(draw_view)

    @app.get("/")
    def index() -> Response:
        return app.send_static_file("index.html")

    @app.get("/healthz")
    def healthz() -> tuple[Response, int]:
        snapshot = pipeline.metrics.snapshot()
        age = snapshot["last_frame_age_s"]
        healthy = snapshot["state"] == "running" and age is not None and age < STALE_AFTER_S
        body = {
            "status": "ok" if healthy else "degraded",
            "state": snapshot["state"],
            "last_frame_age_s": age,
        }
        return jsonify(body), 200 if healthy else 503

    @app.get("/metrics")
    def metrics() -> Response:
        return jsonify({**pipeline.metrics.snapshot(), "waiting": pipeline.waiting})

    @app.get("/tracks")
    def tracks() -> tuple[Response, int]:
        result = pipeline.results.latest
        if result is None:
            return jsonify({"error": "no result yet"}), 404
        return jsonify(result.to_dict()), 200

    @app.get("/events")
    def events() -> Response:
        def stream() -> Iterator[str]:
            with pipeline.results.subscription() as slot:
                while not slot.closed:
                    result = slot.get(timeout=1.0)
                    if result is None:
                        yield ": keep-alive\n\n"  # lets the server notice a closed tab
                    else:
                        yield f"data: {json.dumps(result.to_dict())}\n\n"

        headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        return Response(stream(), mimetype="text/event-stream", headers=headers)

    def mjpeg(jpegs: JpegCache) -> Response:
        def stream() -> Iterator[bytes]:
            with pipeline.results.subscription() as slot:
                while not slot.closed:
                    result = slot.get(timeout=1.0)
                    if result is None:
                        continue
                    jpeg = jpegs.get(result)
                    yield (
                        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(jpeg)).encode()
                        + b"\r\n\r\n"
                        + jpeg
                        + b"\r\n"
                    )

        return Response(stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.get("/video.mjpg")
    def video() -> Response:
        return mjpeg(frames)

    @app.get("/follow.mjpg")
    def follow_video() -> Response:
        return mjpeg(views)

    @app.post("/follow")
    def follow() -> tuple[Response, int]:
        point = request.get_json(silent=True)
        try:
            x, y = float(point["x"]), float(point["y"])
        except (TypeError, KeyError, ValueError):
            return jsonify({"error": 'expected {"x": number, "y": number}'}), 400
        pipeline.follower.lock_at(x, y)
        return jsonify({"x": x, "y": y}), 200

    @app.get("/controls")
    def get_controls() -> Response:
        if controls is None:
            return jsonify({"enabled": False})
        return jsonify({"enabled": True, **controls.state()})

    @app.post("/controls")
    def set_controls() -> tuple[Response, int]:
        if controls is None:
            return jsonify({"error": "start the service with --controls to allow this"}), 403
        changes = request.get_json(silent=True)
        if not isinstance(changes, dict):
            return jsonify({"error": "expected a JSON object"}), 400
        try:
            controls.apply(changes)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"enabled": True, **controls.state()}), 200

    return app
