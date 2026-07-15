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
    monkeypatch.setattr(session_module, "RMVPE", _FakeRuntime)
    path = tmp_path / "weights.pt"
    path.write_bytes(b"weights")
    session = RMVPESession(model_path=path)
    assert session.status == "new"
    with pytest.raises(RuntimeError):
        session.infer([1])
    session.load()
    assert session.status == "ready"
    assert session.infer([1]) == [1]
    session.release()
    assert session.status == "released"
    with pytest.raises(RuntimeError):
        session.infer([1])
    session.close()
    assert session.status == "closed"
    session.close()


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
