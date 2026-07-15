"""Package-owned checkpoint configuration and generic overrides.

Reads: config/checkpoints.toml.  Runtime configuration stays package-local so
callers can override URLs, checksums, and cache locations without a global
registry.
"""

from __future__ import annotations

from pathlib import Path
import tomllib

_CONFIG_PATH = Path(__file__).with_name("config") / "checkpoints.toml"


def checkpoint_config(path: str | Path | None = None) -> dict:
    """Load checkpoint metadata from the package config or an override file."""
    config_path = Path(path) if path is not None else _CONFIG_PATH
    with config_path.open("rb") as stream:
        data = tomllib.load(stream)
    checkpoints = data.get("checkpoints")
    if not isinstance(checkpoints, dict) or not checkpoints:
        raise ValueError(f"{config_path}: missing non-empty [checkpoints] table")
    return checkpoints


def checkpoint_entry(model: str = "rmvpe", *, config_path: str | Path | None = None,
                     overrides: dict | None = None) -> dict:
    """Return one checkpoint entry with generic caller overrides applied."""
    entries = checkpoint_config(config_path)
    if model not in entries:
        raise KeyError(f"unknown RMVPE checkpoint: {model}")
    entry = dict(entries[model])
    if overrides:
        entry.update(overrides)
    return entry
