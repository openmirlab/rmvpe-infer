"""Independent RMVPE model lifecycle session."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import checkpoint_entry
from .download import download_model, resolve_model_path
from .inference import RMVPE


class RMVPESession:
    """Explicit load/infer/release lifecycle for one RMVPE checkpoint."""

    def __init__(self, model_path: str | Path | None = None, *, device=None,
                 hop_length: int = 160, cache_dir=None, model: str = "rmvpe",
                 config_path=None, checkpoint_overrides: dict | None = None):
        self.model_path = Path(model_path) if model_path is not None else None
        self.device = device
        self.hop_length = hop_length
        self.cache_dir = cache_dir
        self.model = model
        self.config_path = config_path
        self.checkpoint_overrides = checkpoint_overrides
        self._runtime: RMVPE | None = None
        self._status = "new"

    @property
    def status(self) -> str:
        return self._status

    def load(self) -> "RMVPESession":
        if self._status == "closed":
            raise RuntimeError("cannot load a closed RMVPE session")
        if self._runtime is not None:
            return self
        try:
            entry = checkpoint_entry(self.model, config_path=self.config_path,
                                     overrides=self.checkpoint_overrides)
            path = self.model_path or download_model(
                cache_dir=self.cache_dir, url=entry["url"], expected_sha256=entry["sha256"],
                filename=entry.get("filename", "model.pt"))
            self._runtime = RMVPE(path, hop_length=self.hop_length, device=self.device)
        except Exception:
            self._status = "failed"
            raise
        self.model_path = Path(path)
        self._status = "ready"
        return self

    def infer(self, audio, **kwargs):
        if self._runtime is None or self._status != "ready":
            raise RuntimeError("RMVPE session is not ready; call load() first")
        return self._runtime.infer_from_audio(audio, **kwargs)

    def release(self) -> "RMVPESession":
        if self._status == "closed":
            return self
        if self._runtime is not None:
            self._runtime.model = None
            self._runtime.mel_extractor = None
            self._runtime.resample_kernel.clear()
            self._runtime = None
        if self._status != "closed":
            self._status = "released"
        return self

    def close(self) -> "RMVPESession":
        if self._status != "closed":
            self.release()
            self._status = "closed"
        return self

    def cache_info(self) -> dict[str, Any]:
        entry = checkpoint_entry(self.model, config_path=self.config_path,
                                 overrides=self.checkpoint_overrides)
        path = self.model_path or resolve_model_path(
            self.cache_dir, filename=entry.get("filename", "model.pt")
        )
        return {"path": str(path) if path else None, "exists": bool(path and path.exists()),
                "cache_dir": str(self.cache_dir) if self.cache_dir else None,
                "model": self.model, "url": entry["url"], "sha256": entry["sha256"],
                "status": self.status, "loaded": self._runtime is not None}

    def __enter__(self):
        return self.load()

    def __exit__(self, exc_type, exc, tb):
        self.close()
