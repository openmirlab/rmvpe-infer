"""Lifecycle and package-owned checkpoint configuration tests."""

import pytest

from rmvpe_infer import RMVPESession
from rmvpe_infer import session as session_module


class _FakeRuntime:
    def __init__(self, path, **kwargs):
        self.model = object()
        self.mel_extractor = object()
        self.resample_kernel = {}

    def infer_from_audio(self, audio, **kwargs):
        return audio


def test_session_requires_load(monkeypatch, tmp_path):
    constructed = []

    def build(*args, **kwargs):
        runtime = _FakeRuntime(*args, **kwargs)
        constructed.append(runtime)
        return runtime

    monkeypatch.setattr(session_module, "RMVPE", build)
    path = tmp_path / "weights.pt"
    path.write_bytes(b"weights")
    session = RMVPESession(model_path=path)
    assert session.status == "new"
    with pytest.raises(RuntimeError):
        session.infer([1])
    session.load()
    session.load()
    assert session.status == "ready"
    assert len(constructed) == 1
    assert session.infer([1]) == [1]
    assert session.release() is session
    assert session.status == "released"
    with pytest.raises(RuntimeError):
        session.infer([1])
    session.load()
    assert len(constructed) == 2
    assert session.close() is session
    assert session.status == "closed"
    assert session.close() is session
    with pytest.raises(RuntimeError, match="closed"):
        session.load()
    with pytest.raises(RuntimeError, match="ready"):
        session.infer([1])


def test_session_failed_load_is_visible(monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise ValueError("bad checkpoint")

    monkeypatch.setattr(session_module, "RMVPE", fail)
    path = tmp_path / "weights.pt"
    path.write_bytes(b"weights")
    session = RMVPESession(model_path=path)
    with pytest.raises(ValueError):
        session.load()
    assert session.status == "failed"


def test_context_manager_releases(monkeypatch, tmp_path):
    monkeypatch.setattr(session_module, "RMVPE", _FakeRuntime)
    path = tmp_path / "weights.pt"
    path.write_bytes(b"weights")
    with RMVPESession(model_path=path) as session:
        assert session.status == "ready"
    assert session.status == "closed"


def test_cache_info_is_read_only_and_uses_default_or_custom_resolver(monkeypatch, tmp_path):
    import rmvpe_infer.download as download

    cache = tmp_path / "cache"
    monkeypatch.delenv(download.ENV_VAR, raising=False)
    monkeypatch.setattr(download, "urlretrieve", lambda *args: pytest.fail("download attempted"))
    default = RMVPESession(cache_dir=cache)
    info = default.cache_info()
    assert info["path"] == str(cache / "model.pt")
    assert info["exists"] is False
    assert not cache.exists()

    custom = RMVPESession(
        cache_dir=cache,
        checkpoint_overrides={
            "url": "https://example.invalid/custom.zip",
            "sha256": "f" * 64,
            "filename": "custom.pt",
        },
    )
    custom_info = custom.cache_info()
    assert custom_info["path"] == str(cache / "custom.pt")
    assert custom_info["url"] == "https://example.invalid/custom.zip"
    assert custom_info["sha256"] == "f" * 64


def test_load_uses_the_same_custom_checkpoint_metadata(monkeypatch, tmp_path):
    captured = {}
    path = tmp_path / "custom.pt"
    path.write_bytes(b"weights")
    monkeypatch.setattr(session_module, "RMVPE", _FakeRuntime)
    monkeypatch.setattr(
        session_module,
        "download_model",
        lambda **kwargs: captured.update(kwargs) or path,
    )
    session = RMVPESession(
        cache_dir=tmp_path / "cache",
        checkpoint_overrides={
            "url": "https://example.invalid/custom.zip",
            "sha256": "a" * 64,
            "filename": "custom.pt",
        },
    )
    session.load()
    assert captured == {
        "cache_dir": tmp_path / "cache",
        "url": "https://example.invalid/custom.zip",
        "expected_sha256": "a" * 64,
        "filename": "custom.pt",
    }
