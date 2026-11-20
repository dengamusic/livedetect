# livedetect

A small real-time video service. It reads a camera stream, detects and tracks
drones, and serves the annotated video, the tracks and its own health over
HTTP. The camera can be slower or faster than the detector, and it can
disappear; the service stays live through both.

![A drone tracked by the service](docs/demo.gif)

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

This starts two containers: a stand-in network camera that plays a drone clip
as H.264 over TCP, and the service reading from it. To see the service lose
and regain its camera:

```bash
docker compose stop camera     # page shows "reconnecting", /healthz returns 503
docker compose start camera    # back to "running" at the next retry, 8 s at most
```

Other things the stand-in camera can show:

```bash
CAMERA_SOURCE=cv:/app/assets/drone_far.mp4 docker compose up        # a small, distant drone
CAMERA_SOURCE=cv:/clips/yourclip.mp4 docker compose up              # your own clip, from ./clips
CAMERA_SOURCE=synthetic LIVEDETECT_DETECTOR=motion docker compose up   # generated scene, no model
```

Without Docker:

```bash
python3 -m venv --system-site-packages .venv && . .venv/bin/activate
pip install -e ".[dev,onnx]"
python -m livedetect --source assets/drone_close.mp4 --detector drone
python -m livedetect                                   # generated scene, no model, no files
python -m livedetect --source rtsp://user:pass@camera/stream --detector drone
```

`--system-site-packages` lets the environment see GStreamer's Python bindings,
which come from the system (`apt install python3-gi gir1.2-gstreamer-1.0
gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-libav`). The
generated scene and `cv:` sources work without them.

## Sources

| `--source`                     | Reads                                                     |
|--------------------------------|-----------------------------------------------------------|
| `synthetic`                    | A generated sky with three small moving objects           |
| `rtsp://…`, `file://…`, a path | Anything GStreamer's `uridecodebin` can open              |
| `gst:<pipeline>`               | The front half of a GStreamer pipeline you write yourself |
| `cv:<path, URL or index>`      | OpenCV, for machines without GStreamer                    |

For a low-latency RTSP camera, write the pipeline out so you control the
jitter buffer: `gst:rtspsrc location=rtsp://… latency=100 ! decodebin`.

## Detectors

| `--detector`            | What it is                                                              | Needs          |
|-------------------------|-------------------------------------------------------------------------|----------------|
| `motion`                | Background subtraction: anything that moves in front of a fixed camera   | nothing        |
| `drone`                 | A YOLOv5s network trained on drones, in ONNX, run with onnxruntime       | `.[onnx]`      |
| `ssdlite`, `fasterrcnn` | General-purpose PyTorch networks (people, cars, birds, planes)           | `.[torch]`     |

The pipeline only sees the `Detector` interface (`detect(image) -> detections`),
so another model is one more class in `detectors.py`.

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
onnxruntime, PyTorch) release the interpreter lock, so two threads run in
parallel where it matters, and frames are passed by reference instead of being
copied between processes. Several cameras on one machine would be one process
per camera.

**Interfaces at the two places that change.** `FrameSource` and `Detector` are
three-line protocols. The pipeline is tested against fakes that fail on a
schedule, so reconnects, backoff and frame dropping are covered without a
camera, a model or a GPU.

**Model files are pinned and verified, not committed.** `models.py` names an
exact revision of the weights and the SHA-256 they must have, downloads them
once into a cache, and refuses a file that does not match. The Docker build
does this at build time, so the container runs without network access.

**Latency is measured from the moment a frame enters the process** to the
moment its result is ready. It does not include the camera, the network or the
video decoder, which this process cannot see.

## Tests

```bash
pytest          # about ten seconds
ruff check . && ruff format --check .
```

The GStreamer tests are skipped where the bindings are missing, the PyTorch
test where `torch` is missing, and the drone-model test until the model is in
the cache (`python -m livedetect.models`).

On every push, GitHub Actions (`.github/workflows/ci.yml`) runs lint and the
tests, then builds the Docker image, starts both containers, checks that a
drone is being tracked, stops the camera and checks that the service reports
the loss and recovers.

## Limits

- The drone model is a small community model. It finds the drone in nearly
  every frame of the two bundled clips; that says nothing about birds,
  clutter, long range or night, which is where drone detection is hard.
- The motion detector assumes a fixed camera and fires on anything that moves.
- The tracker matches greedily by distance. Many close targets would need
  optimal assignment and a proper Kalman filter.
- One camera per process, and Werkzeug's threaded server. Fine for a handful
  of viewers; more would want a production WSGI server in front.
- Stopping can take up to the source's 3 s read timeout if the camera is
  already dead.

## Credits

- Demo clips: Wikimedia Commons, by NaIzletuSi (CC BY 3.0) and Quahadi
  (CC BY-SA 3.0). Details and changes in [assets/CREDITS.md](assets/CREDITS.md).
- Drone model: [engdarwish/drone-detection-yolov5](https://huggingface.co/engdarwish/drone-detection-yolov5)
  on Hugging Face, MIT licence.
