"""Shared fixtures for rmvpe_infer tests.

Provides synthetic-audio helpers (pure tones, a vibrato sweep, a seeded
noisy-vocal-ish clip) reused by the physics accuracy test and the golden
regression fixture, plus a `weights_path` fixture that locates a real
checkpoint (via `RMVPE_INFER_WEIGHTS` or the default `~/.cache/rmvpe/`
cache) WITHOUT downloading anything -- tests marked `weights` skip cleanly
when no checkpoint is present rather than triggering a network fetch.

Reads: rmvpe_infer.download (DEFAULT_CACHE_DIR, ENV_VAR).
"""

import os
from pathlib import Path

import numpy as np
import pytest

from rmvpe_infer.download import DEFAULT_CACHE_DIR, ENV_VAR

SAMPLE_RATE = 16000


def find_local_weights():
    """Return a Path to an already-present checkpoint, or None. Never downloads."""
    env_path = os.environ.get(ENV_VAR)
    if env_path and Path(env_path).exists():
        return Path(env_path)
    for name in ("model.pt", "rmvpe.pt"):
        candidate = DEFAULT_CACHE_DIR / name
        if candidate.exists():
            return candidate
    return None


@pytest.fixture(scope="session")
def weights_path():
    """Path to a real RMVPE checkpoint; skips the test if none is available locally."""
    path = find_local_weights()
    if path is None:
        pytest.skip(
            f"no local RMVPE checkpoint found (set {ENV_VAR} or populate "
            f"{DEFAULT_CACHE_DIR}) -- skipping weight-dependent test"
        )
    return path


@pytest.fixture
def random_audio():
    """1 second of reproducible random audio at 16kHz."""
    rng = np.random.default_rng(42)
    return rng.standard_normal(SAMPLE_RATE).astype(np.float32)


def sine_tone(freq: float, duration: float = 1.0, sr: int = SAMPLE_RATE, amplitude: float = 0.5):
    """A pure sine wave at `freq` Hz -- the simplest possible ground-truth F0 signal."""
    t = np.arange(int(sr * duration)) / sr
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def vibrato_tone(
    carrier: float = 440.0,
    depth: float = 20.0,
    rate: float = 5.0,
    duration: float = 1.0,
    sr: int = SAMPLE_RATE,
    amplitude: float = 0.5,
):
    """A frequency-modulated tone: instantaneous freq = carrier + depth*sin(2*pi*rate*t).

    Returns (audio, instantaneous_freq_per_sample) so callers can resample the
    ground-truth curve to whatever hop-length frame times they decode at.
    """
    t = np.arange(int(sr * duration)) / sr
    inst_freq = carrier + depth * np.sin(2 * np.pi * rate * t)
    phase = 2 * np.pi * np.cumsum(inst_freq) / sr
    audio = (amplitude * np.sin(phase)).astype(np.float32)
    return audio, inst_freq


def seeded_noisy_vocal_clip(sr: int = SAMPLE_RATE, duration: float = 1.0, seed: int = 1234):
    """A deterministic synthetic "vocal-ish" clip: a harmonic series (fundamental +
    4 decaying overtones, roughly voice-like) plus a touch of seeded noise.

    Not a real recording -- a fully reproducible stand-in so the golden
    regression fixture doesn't need to commit or fetch an audio asset.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * duration)) / sr
    carrier = 180.0
    harmonics = sum(0.5 / k * np.sin(2 * np.pi * carrier * k * t) for k in range(1, 6))
    noise = 0.02 * rng.standard_normal(len(t))
    audio = (harmonics + noise).astype(np.float32)
    audio = audio / np.max(np.abs(audio)) * 0.7
    return audio.astype(np.float32)
