# RMVPE-Infer

**Production-ready, inference-only toolkit for robust vocal pitch estimation in polyphonic music**

RMVPE-Infer provides a clean, lightweight API for running vocal pitch (F0) estimation using the RMVPE model with automatic checkpoint management.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Why this exists

[RMVPE](https://github.com/yxlllc/RMVPE) is a deep U-Net + BiGRU model for
robust vocal pitch (F0) estimation directly from polyphonic mixes, published
at INTERSPEECH 2023. The reference implementation is a research repo: it
ships training code, several loosely-related sub-projects, and a pretrained
checkpoint distributed as a `.zip` attached to a GitHub release rather than
a package on PyPI. It has since become best known as the pitch-extraction
backbone quietly vendored inside voice-conversion projects (e.g. RVC), which
means the "just run inference" path is buried under code most callers don't
need.

RMVPE-Infer reprovides just that inference path: a small, pip-installable,
inference-only package with a single public class (`RMVPE`), an
auto-downloading checkpoint fetcher with sha256 verification, and a CLI —
nothing else from the original repo.

---

## Acknowledgments

This project wraps, and would not exist without, the original research and
implementation:

- **[yxlllc/RMVPE](https://github.com/yxlllc/RMVPE)** — the reference
  implementation this package wraps; also the current host of the pretrained
  checkpoint (see [Pretrained Model](#pretrained-model) below).
- **Haojie Wei, Xueke Cao, Tangpeng Dan, Yueguo Chen** — authors of the
  RMVPE paper (see [Citation](#citation)).
- **[INTERSPEECH 2023](https://doi.org/10.21437/Interspeech.2023-528)** —
  the paper's publication venue (Dublin, Ireland, August 2023).

---

## Citation

If you use RMVPE-Infer in your research, please cite the original paper:

```bibtex
@article{wei2023rmvpe,
    title   = {RMVPE: A Robust Model for Vocal Pitch Estimation in Polyphonic Music},
    author  = {Wei, Haojie and Cao, Xueke and Dan, Tangpeng and Chen, Yueguo},
    journal = {arXiv preprint arXiv:2306.15412},
    year    = {2023}
}
```

The paper was later published at INTERSPEECH 2023 (pp. 5421-5425,
[doi:10.21437/Interspeech.2023-528](https://doi.org/10.21437/Interspeech.2023-528));
cite whichever version matches your bibliography style.

---

## Features

- **Inference Only**: Lightweight package focused on production inference
- **Auto-Download**: Automatic pretrained checkpoint download (~340 MB), sha256-verified
- **Polyphonic-Robust**: Extracts vocal pitch directly from mixed audio without source separation
- **GPU Accelerated**: Full CUDA support with automatic device detection
- **CLI Tool**: `rmvpe-infer` command for quick pitch extraction
- **Python API**: Clean programmatic interface

---

## Scope

**In scope:** a single-model, inference-only wrapper around RMVPE — load a
checkpoint, run F0 estimation on an audio buffer, get back per-frame pitch
in Hz. That's the entire public surface (`RMVPE`, `download_model`).

**Out of scope, forever:**
- Training or fine-tuning RMVPE (this package never loads a dataset or computes a loss)
- Other pitch-estimation models (CREPE, pYIN, etc.) — this is RMVPE-only by design
- Source separation / stem splitting (RMVPE works directly on polyphonic mixes; that's the point)
- Voice conversion or any downstream use of the extracted pitch (e.g. RVC-style pipelines) — those are separate projects that may *consume* this package's output, not something this package does

---

## Install

```bash
# Using pip
pip install rmvpe-infer

# Using UV (recommended)
uv pip install rmvpe-infer
```

### Development Installation

```bash
# Clone repository
git clone https://github.com/openmirlab/rmvpe-infer.git
cd rmvpe-infer

# Install with UV
uv sync --extra dev

# Install with pip
pip install -e ".[dev]"
```

---

## Quick Start

### CLI Inference

```bash
# Basic usage (auto-downloads model on first run)
rmvpe-infer -i vocals.wav -o f0.csv

# With options
rmvpe-infer -i vocals.wav -o f0.csv --threshold 0.05 --viterbi --device cuda
```

### Python API

```python
from rmvpe_infer import RMVPE, download_model

# Download pretrained model (cached after first call)
model_path = download_model()

# Load model (auto-detects GPU)
rmvpe = RMVPE(str(model_path))

# Run inference
import librosa
audio, sr = librosa.load("vocals.wav", sr=None, mono=True)
f0 = rmvpe.infer_from_audio(audio, sample_rate=sr)
# f0: numpy array of F0 values in Hz (0 = unvoiced)
```

**Sanity-check it on a known pitch** (the same tones the test suite verifies
against in `tests/test_pitch_physics.py`):

```python
import numpy as np
from rmvpe_infer import RMVPE, download_model

rmvpe = RMVPE(str(download_model()))

sr = 16000
t = np.arange(sr) / sr  # 1 second
audio = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)  # A4, 440Hz

f0 = rmvpe.infer_from_audio(audio, sample_rate=sr)
voiced = f0[f0 > 0]
print(f"median detected pitch: {np.median(voiced):.1f} Hz (expected ~440 Hz)")
```

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hop_length` | `160` | Hop size in samples at 16kHz (160 = 10ms frames) |
| `device` | auto | `"cuda"`, `"cpu"`, or `None` for auto-detect |
| `thred` | `0.03` | Voicing confidence threshold |
| `use_viterbi` | `False` | Use Viterbi decoding for smoother pitch tracks |

---

## Pretrained Model

The pretrained checkpoint is automatically downloaded from the [official RMVPE release](https://github.com/yxlllc/RMVPE/releases/tag/230917) (~350 MB extracted `.pt`, from `rmvpe.zip`).

- **Architecture**: Deep U-Net encoder-decoder + BiGRU
- **Training data**: MIR-1K, PTDB, M4Singer
- **Frame rate**: 10ms (100 fps at 16kHz)
- **Frequency range**: ~30 Hz to 8000 Hz
- **Cache location**: `~/.cache/rmvpe/`
- **Provenance**: sha256 of the extracted checkpoint is
  `19dc1809cf4cdb0a18db93441816bc327e14e5644b72eeaae5220560c6736fe2`,
  verified automatically on every download and cache hit
  (`rmvpe_infer.download.verify_checksum`) — a corrupted or tampered file
  raises `ChecksumMismatchError` instead of loading silently.
- **Hosting note**: this URL is yxlllc's own third-party GitHub release, not
  currently an openmirlab-controlled mirror. Set `RMVPE_INFER_WEIGHTS=/path/to/checkpoint.pt`
  to point at your own copy and skip the download entirely (also useful for
  offline/air-gapped environments).

---

## Testing

```bash
# CI-safe unit tests — no checkpoint, no network, no GPU (31 tests)
uv run pytest tests/

# Weight-dependent tests — needs the real checkpoint (auto-downloads if
# not cached, or set RMVPE_INFER_WEIGHTS to point at your own copy):
# pitch-accuracy physics test (220/440/880Hz + vibrato sweep vs ground
# truth), a CPU-determinism check, and a golden regression fixture.
uv run pytest tests/ -m weights
```

See [CLAUDE.md](CLAUDE.md) for the full test-layer breakdown.

---

## What this project will NEVER bundle

RMVPE-Infer downloads a pretrained checkpoint at runtime — it does **not**,
and will never, ship model weights inside the pip package or the git
repository itself:

- The `.pt` checkpoint (~340 MB) is fetched on first use from the upstream
  release URL (or your own copy via `RMVPE_INFER_WEIGHTS`) into
  `~/.cache/rmvpe/`, never committed to this repo or bundled into the wheel.
- Every download (and every cache hit) is sha256-verified against a pinned
  hash before it's allowed to load — a truncated, corrupted, or
  silently-swapped file raises `ChecksumMismatchError` instead of loading.
- The current download URL is a third-party GitHub release (yxlllc's, not
  an openmirlab-controlled mirror). This is a known gap — the project's
  intent is to eventually host weights under org control — tracked, not
  bundled around.

---

## Development

Uses [`uv`](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install dependencies
uv sync --extra dev

# Run the CI-safe test suite
uv run pytest tests/

# Lint
uv run ruff check .
```

Package version is single-sourced in `src/rmvpe_infer/__about__.py` — don't
hand-edit a version literal in `pyproject.toml` or `__init__.py` directly.
See [CLAUDE.md](CLAUDE.md) for the full test-layer breakdown and dev workflow notes.

---

## License

MIT License - see [LICENSE](LICENSE) for details.

This project includes code adapted from **RMVPE** by yxlllc (see
Acknowledgments above).

---

## Support

For issues and questions:
- **GitHub Issues**: [github.com/openmirlab/rmvpe-infer/issues](https://github.com/openmirlab/rmvpe-infer/issues)

---
