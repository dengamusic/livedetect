# livedetect

A small real-time video service. It reads a camera stream, detects and tracks
drones, steers a tracking camera onto one of them, and serves the video, the
tracks and its own health over HTTP. The camera can be slower or faster than
the detector, and it can disappear; the service stays live through both.

![Left: the whole frame with the tracked drone. Right: the tracking camera's view of it.](docs/demo.gif)

```
            capture thread                    process thread                              HTTP threads
 camera ──► source.read() ──► [newest frame] ──► detector ──► tracker ──► tracking ──► [broadcast] ──► /video.mjpg
            reconnects with     one slot,        (+ tiles)                camera        one slot per    /follow.mjpg
            backoff             overwrites                                              viewer          /events
```

## Run it

With Docker (nothing else needed):

```bash
docker compose up --build
# open http://localhost:8000
```

This starts two containers: a stand-in network camera that plays a drone clip
as H.264 over TCP, and the service reading from it.

Without Docker:

```bash
python3 -m venv --system-site-packages .venv && . .venv/bin/activate
pip install -e ".[dev,onnx]"
python -m livedetect --source cv:assets/drone_far.mp4 --detector drone --controls
python -m livedetect                                   # generated scene, no model, no files
python -m livedetect --source rtsp://user:pass@camera/stream --detector drone
```

`--system-site-packages` lets the environment see GStreamer's Python bindings,
which come from the system (`apt install python3-gi gir1.2-gstreamer-1.0
gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-libav`). The
generated scene and `cv:` sources work without them.

## Things to try

`--controls` (set in the Docker setup) puts four switches on the page. They
are off by default: a button that cuts the camera does not belong on a service
that is watching something.

![The live page, with the detector slowed down and the queue switched on](docs/page.png)

1. **Click a target** in the main picture. The tracking camera on the right
   pans to it and zooms until it fills a quarter of the view's height.
2. **Slow the detector** by 300 ms. Results fall to 3 a second, the dropped
   count climbs, and latency settles at about 0.34 s: one detector run, no more.
3. **Switch the hand-over to a queue** with the detector still slow. Latency
   climbs to about 9.8 s and stays there: 30 frames waiting, 0.3 s each. Switch
   back and it is 0.34 s again. This is why the pipeline drops frames.
4. **Cut the camera.** The badge turns to "reconnecting" and `/healthz` returns
   503. Connect it again and the stream is back at the next retry, 8 s at most.
   With Docker, `docker compose stop camera` does the same to a real connection.
5. **Turn on tiles** with a drone too small for the network:

   ```bash
   CAMERA_SOURCE=cv:/app/assets/drone_tiny.mp4 docker compose up
   python -m livedetect --source cv:assets/drone_tiny.mp4 --detector drone --controls
   ```

   Without tiles the drone, 7 pixels wide, is found in about half of the
   frames and hardly at all in the first six seconds of the clip. With 3 × 2
   tiles it is found in nine frames out of ten, and the charts show what that
   costs.

Other clips for the stand-in camera:

```bash
CAMERA_SOURCE=cv:/app/assets/drone_close.mp4 docker compose up      # a large drone against blue sky
CAMERA_SOURCE=cv:/clips/yourclip.mp4 docker compose up              # your own clip, from ./clips
CAMERA_SOURCE=synthetic LIVEDETECT_DETECTOR=motion docker compose up   # generated scene, no model
```

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

`--tiles 3x2` also runs the network on a grid of overlapping tiles; see
[Tiny targets](#tiny-targets). The pipeline only sees the `Detector` interface
(`detect(image) -> detections`), so another model is one more class in
`detectors.py`.

## HTTP interface

| Route              | Returns                                                         |
|--------------------|-----------------------------------------------------------------|
| `GET /`            | Live page: both pictures, charts of the last minute, switches   |
| `GET /healthz`     | 200 while frames are flowing, 503 otherwise                     |
| `GET /metrics`     | Frame rates in and out, latency percentiles, drops, reconnects  |
| `GET /tracks`      | The most recent result as JSON                                  |
| `GET /events`      | Results as they happen (server-sent events)                     |
| `GET /video.mjpg`  | The whole frame with tracks drawn on it (MJPEG)                 |
| `GET /follow.mjpg` | The tracking camera's view (MJPEG)                              |
| `POST /follow`     | `{"x": 480, "y": 270}`: follow the track nearest to that point  |
| `GET /controls`    | The state of the switches                                       |
| `POST /controls`   | `{"detector_delay_ms": 300}`, `{"camera_cut": true}`, `{"queue": true}`, `{"tiles": "3x2"}`; 403 without `--controls` |

## Tracking camera

A real tracking camera sits on a motorised mount with a zoom lens. Here the
mount is a window that moves over the frame, and the zoom is the size of that
window (`follow.py`). The control law is the one a mount needs:

    pan rate = error / time constant + target velocity,   limited to a top speed

The first term closes the distance to the target, the second keeps up with a
target that is moving, and the limit makes the picture pan instead of jump.
The gain is computed from the real time between frames, so a slow frame rate
makes the camera sluggish but cannot make it overshoot. It stays on its target
while the tracker is only predicting it, waits a second after losing it, then
zooms back out. The view is resampled through one affine map, so the window
moves by fractions of a pixel and the enlarged picture does not judder.

## Tiny targets

The drone model looks at its input at 416 pixels. A 960-pixel frame is shrunk
to less than half for it, so a drone 12 pixels wide arrives 5 pixels wide.
With `--tiles 3x2` the network also runs on six overlapping parts of the
frame, each close to its real resolution, and the results are merged.

`bench/tiny_targets.py` measures what that buys. There is no labelled footage
of tiny drones here, so it makes its own: it takes frames where the detector is
sure, paints the drone out, and pastes it back smaller, as if it were further
away. Share of frames in which the drone was found:

| Drone width (px) | plain | 3 × 2 tiles | 5 × 3 tiles |
|-----------------:|------:|------------:|------------:|
| 41 to 14         | 100 % | 100 %       | 100 %       |
| 10               | 85 %  | 100 %       | 100 %       |
| 8                | 67 %  | 98 %        | 95 %        |
| 7                | 55 %  | 93 %        | 81 %        |
| 5                | 41 %  | 81 %        | 60 %        |

|                                   | plain | 3 × 2 tiles | 5 × 3 tiles |
|-----------------------------------|------:|------------:|------------:|
| Network runs per frame            | 1     | 7           | 16          |
| Time per frame (ms, 4 CPU threads) | 18    | 127         | 289         |
| False alarms per 100 frames       | 0.0   | 0.0         | 3.1         |

3 × 2 tiles roughly halve the size at which the drone is still found, at seven
times the compute. Finer tiles are worse on every row: a 5 × 3 tile is already
shown larger than it was recorded, so the network gets no new detail, only
more chances to be wrong.

These numbers come from 129 frames of one clip with the drone against open
sky. Haze, motion blur and clutter behind the target are not in them, and all
three make real range harder.

## Design decisions

**Drop frames, never queue them.** Capture and processing run in separate
threads joined by a one-item slot that overwrites (`buffers.LatestSlot`). When
the detector is slower than the camera, it always works on the newest frame.
With a queue, every frame waits behind the ones before it, and each result is
a full queue of processing times old. `buffers.FifoQueue` is that queue; the
pipeline uses it only when the switch on the page asks for the comparison.

**A slow viewer cannot slow the pipeline.** Each HTTP client gets its own
overwrite slot (`buffers.Broadcast`). A browser on a bad connection skips
frames; the pipeline and the other viewers do not notice. Frames are drawn and
encoded once, however many viewers there are.

**The camera is expected to fail.** Any error or end of stream from the source
closes it and reopens it: immediately if it had been delivering frames, then
with exponential backoff up to 8 s. The same path makes a video file loop.
After a gap longer than a second the tracker discards its tracks, because
predictions across the gap would be wrong.

**Limits are in seconds, not frames.** The frame rate is not a constant: it
falls whenever the detector is slower than the camera. The tracker therefore
drops a track after half a second without a detection, widens its matching
distance with the time since the last frame, and corrects speed over the time
since the target was last seen. Counted in frames, each of these was right at
30 frames a second and wrong at 5.

**Threads, not processes or asyncio.** The heavy calls (GStreamer, OpenCV,
onnxruntime, PyTorch) release the interpreter lock, so two threads run in
parallel where it matters, and frames are passed by reference instead of being
copied between processes. Several cameras on one machine would be one process
per camera.

**Interfaces at the two places that change.** `FrameSource` and `Detector` are
three-line protocols. The pipeline is tested against fakes that fail on a
schedule, so reconnects, backoff and frame dropping are covered without a
camera, a model or a GPU. Tiling and the fault switches are wrappers around
the same two interfaces; the pipeline does not know about either.

**Model files are pinned and verified, not committed.** `models.py` names an
exact revision of the weights and the SHA-256 they must have, downloads them
once into a cache, and refuses a file that does not match. The Docker build
does this at build time, so the container runs without network access.

**Latency is measured from the moment a frame enters the process** to the
moment its result is ready. It does not include the camera, the network or the
video decoder, which this process cannot see.

## Tests

```bash
pytest          # about fifteen seconds
ruff check . && ruff format --check .
```

The GStreamer tests are skipped where the bindings are missing, the PyTorch
test where `torch` is missing, and the two drone-model tests until the model
is in the cache (`python -m livedetect.models`).

On every push, GitHub Actions (`.github/workflows/ci.yml`) runs lint and the
tests, then builds the Docker image, starts both containers, checks that a
drone is being tracked and that the switches answer, stops the camera and
checks that the service reports the loss and recovers.

## Limits

- The drone model is a small community model. It finds the drone in nearly
  every frame of the two original clips; that says nothing about birds,
  clutter, long range or night, which is where drone detection is hard.
- `drone_tiny.mp4` is an edited clip, not footage of a distant drone, and the
  tiling numbers come from the same edit.
- Tiling runs every tile on every frame. Running only the tiles near existing
  tracks, and sweeping the rest in turn, would cut most of the cost.
- The tracking camera moves a window over a fixed picture. A real mount moves
  the picture too, which the tracker would have to compensate for.
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
