"""Model files the service can download: `python -m livedetect.models`.

Weights are not kept in the repository. Each entry names an exact revision of
a public file and the SHA-256 it must have, so every machine gets the same
bytes or an error. Files are cached under ~/.cache/livedetect (or
$LIVEDETECT_CACHE).
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelFile:
    url: str
    sha256: str
    filename: str
    licence: str


MODELS = {
    # YOLOv5s fine-tuned for a single "drone" class, by engdarwish on Hugging Face.
    "drone": ModelFile(
        url="https://huggingface.co/engdarwish/drone-detection-yolov5/resolve/"
        "dbd4a5e295bb81612ca59189fbdcb591d8e608b6/best.onnx",
        sha256="bea92c72f9a37ca53e3b26ab0d51bceeb6e2086fcb996ad614f1e8de5d989c82",
        filename="drone_yolov5s.onnx",
        licence="MIT",
    ),
}


def cache_dir() -> Path:
    default = Path.home() / ".cache" / "livedetect"
    return Path(os.environ.get("LIVEDETECT_CACHE", default))


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(name: str) -> Path:
    """Return the local path of a model, downloading and verifying it if needed."""
    model = MODELS[name]
    path = cache_dir() / model.filename
    if path.exists() and sha256_of(path) == model.sha256:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading %s model from %s", name, model.url)
    # Download next to the target and rename, so a half-finished download is
    # never mistaken for the file.
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as partial:
        with urllib.request.urlopen(model.url, timeout=60) as response:
            for block in iter(lambda: response.read(1 << 20), b""):
                partial.write(block)
    partial_path = Path(partial.name)
    actual = sha256_of(partial_path)
    if actual != model.sha256:
        partial_path.unlink()
        raise RuntimeError(f"{name} model has SHA-256 {actual}, expected {model.sha256}")
    # NamedTemporaryFile makes the file readable by its owner only. Weights are
    # not secret, and the container downloads them as root at build time and
    # reads them as another user at run time.
    partial_path.chmod(0o644)
    partial_path.replace(path)
    return path


if __name__ == "__main__":
    logging.basicConfig(level="INFO", format="%(message)s")
    for model_name in MODELS:
        print(fetch(model_name))
