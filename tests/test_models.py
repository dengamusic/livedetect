import hashlib

import pytest

from livedetect import models
from livedetect.models import ModelFile


@pytest.fixture
def remote(tmp_path, monkeypatch):
    """A fake download: a local file served through a file:// URL, and an empty cache."""
    payload = tmp_path / "remote.bin"
    payload.write_bytes(b"weights")
    monkeypatch.setenv("LIVEDETECT_CACHE", str(tmp_path / "cache"))

    def register(sha256: str) -> None:
        entry = ModelFile(payload.as_uri(), sha256, "toy.bin", "MIT")
        monkeypatch.setitem(models.MODELS, "toy", entry)

    register(hashlib.sha256(b"weights").hexdigest())
    return payload, register


def test_fetch_downloads_then_serves_from_the_cache(remote, tmp_path):
    payload, _ = remote
    path = models.fetch("toy")
    assert path == tmp_path / "cache" / "toy.bin"
    assert path.read_bytes() == b"weights"
    payload.unlink()  # the second call must not need the remote
    assert models.fetch("toy") == path


def test_a_downloaded_model_is_readable_by_other_users(remote):
    # The container downloads the model as root and runs as another user.
    assert models.fetch("toy").stat().st_mode & 0o777 == 0o644


def test_fetch_rejects_a_download_with_the_wrong_checksum(remote, tmp_path):
    _, register = remote
    register("0" * 64)
    with pytest.raises(RuntimeError, match="SHA-256"):
        models.fetch("toy")
    assert list((tmp_path / "cache").iterdir()) == []  # nothing half-written is left behind


def test_fetch_replaces_a_corrupted_cached_file(remote, tmp_path):
    cached = tmp_path / "cache" / "toy.bin"
    cached.parent.mkdir()
    cached.write_bytes(b"truncated")
    assert models.fetch("toy").read_bytes() == b"weights"


def test_every_listed_model_pins_a_revision_and_a_checksum():
    for name, model in models.MODELS.items():
        assert "/resolve/main/" not in model.url, f"{name} follows a moving branch"
        assert len(model.sha256) == 64
