# livedetect

A small real-time video service. It reads a camera stream, detects and tracks
small moving objects, and serves the annotated video, the tracks and its own
health over HTTP. The camera can be slower or faster than the detector, and it
can disappear; the service stays live through both.

```
            capture thread                    process thread                 HTTP threads
 camera ──► source.read() ──► [newest frame] ──► detector ──► tracker ──► [broadcast] ──► /video.mjpg
            reconnects with     one slot,                                  one slot per    /events
            backoff             overwrites                                 viewer          /tracks
```

## Run it

With Docker (nothing else needed):

```bash
docker compose up --build
# open http://localhost:8000
```

This starts two containers: a stand-in network camera that streams H.264 over
TCP, and the service reading from it. To see the service lose and regain its
camera:

```bash
docker compose stop camera     # page shows "reconnecting", /healthz returns 503
docker compose start camera    # back to "running" at the next retry, 8 s at most
```

To use your own clip as the camera, put it in `clips/` and run
`CAMERA_SOURCE=cv:/clips/yourclip.mp4 docker compose up`.

Without Docker:

```bash
python3 -m venv --system-site-packages .venv && . .venv/bin/activate
pip install -e ".[dev]"
python -m livedetect                      # built-in synthetic scene
python -m livedetect --source clip.mp4    # a file, through GStreamer
python -m livedetect --source rtsp://user:pass@camera/stream
```

`--system-site-packages` lets the environment see GStreamer's Python bindings,
which come from the system (`apt install python3-gi gir1.2-gstreamer-1.0
gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-libav`). The
synthetic scene and `cv:` sources work without them.

## Sources

| `--source`                | Reads                                                    |
|---------------------------|----------------------------------------------------------|
| `synthetic`               | A generated sky with three small moving objects          |
| `rtsp://…`, `file://…`, a path | Anything GStreamer's `uridecodebin` can open        |
| `gst:<pipeline>`          | The front half of a GStreamer pipeline you write yourself |
| `cv:<path, URL or index>` | OpenCV, for machines without GStreamer                   |

For a low-latency RTSP camera, write the pipeline out so you control the
jitter buffer: `gst:rtspsrc location=rtsp://… latency=100 ! decodebin`.

## HTTP interface

| Route         | Returns                                                         |
|---------------|-----------------------------------------------------------------|
| `/`           | Live page: video, pipeline numbers, track table                 |
| `/healthz`    | 200 while frames are flowing, 503 otherwise                     |
| `/metrics`    | Frame rates in and out, latency percentiles, drops, reconnects  |
| `/tracks`     | The most recent result as JSON                                  |
| `/events`     | Results as they happen (server-sent events)                     |
| `/video.mjpg` | Annotated video (MJPEG)                                         |

## Design decisions

**Drop frames, never queue them.** Capture and processing run in separate
threads joined by a one-item slot that overwrites (`buffers.LatestSlot`). When
the detector is slower than the camera, it always works on the newest frame.
With a queue, latency would grow for as long as the service runs; here it is
bounded by about two processing times. `/metrics` reports how many frames were
dropped, so the cost is visible.

**A slow viewer cannot slow the pipeline.** Each HTTP client gets its own
overwrite slot (`buffers.Broadcast`). A browser on a bad connection skips
frames; the pipeline and the other viewers do not notice.

**The camera is expected to fail.** Any error or end of stream from the source
closes it and reopens it: immediately if it had been delivering frames, then
with exponential backoff up to 8 s. The same path makes a video file loop.
After a gap longer than a second the tracker discards its tracks, because
predictions across the gap would be wrong.

**Threads, not processes or asyncio.** The heavy calls (GStreamer, OpenCV,
PyTorch) release the interpreter lock, so two threads run in parallel where it
matters, and frames are passed by reference instead of being copied between
processes. Several cameras on one machine would be one process per camera.

**Interfaces at the two places that change.** `FrameSource` and `Detector` are
three-line protocols. The pipeline is tested against fakes that fail on a
schedule, so reconnects, backoff and frame dropping are covered without a
camera, a model or a GPU.

**Latency is measured from the moment a frame enters the process** to the
moment its result is ready. It does not include the camera, the network or the
video decoder, which this process cannot see.

## Tests

```bash
pytest          # about ten seconds
ruff check . && ruff format --check .
```

The GStreamer tests are skipped where the bindings are missing, and the
PyTorch test where `torch` is missing. `.gitlab-ci.yml` runs lint and tests on
every push.

## Limits

- The motion detector assumes a fixed camera and will fire on anything that
  moves, including birds and clouds. It is a stand-in for a trained network.
- The bundled PyTorch detectors use COCO weights, which have no drone class.
- The tracker matches greedily by distance. Many close targets would need
  optimal assignment and a proper Kalman filter.
- One camera per process, and Werkzeug's threaded server. Fine for a handful
  of viewers; more would want a production WSGI server in front.
- Stopping can take up to the source's 3 s read timeout if the camera is
  already dead.
