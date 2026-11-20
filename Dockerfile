FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# GStreamer and its Python bindings come from the distribution, so the virtual
# environment below is created with --system-site-packages to see them.
#   plugins-good: rtsp     plugins-bad: mpeg-ts, h264parse
#   plugins-ugly: x264 encoder (fake camera only)     libav: H.264 decoder
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-gi \
        gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0 \
        gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv --system-site-packages /opt/venv
ENV PATH=/opt/venv/bin:$PATH

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY assets ./assets

# Build with --build-arg WITH_TORCH=1 to include the PyTorch detectors (CPU wheels).
ARG WITH_TORCH=0
RUN pip install --no-cache-dir ".[onnx]" \
    && if [ "$WITH_TORCH" = "1" ]; then \
         pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu; \
       fi

# Fetch the drone model at build time (pinned revision, checksum verified), so
# the container needs no network when it runs.
ENV LIVEDETECT_CACHE=/opt/livedetect-models
RUN python3 -m livedetect.models

RUN useradd --create-home app
USER app

ENV LIVEDETECT_HOST=0.0.0.0
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=15s CMD \
    python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"

ENTRYPOINT ["python3", "-m", "livedetect"]
