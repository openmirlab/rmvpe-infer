# RMVPE-Infer

**Production-ready, inference-only toolkit for robust vocal pitch estimation in polyphonic music**

RMVPE-Infer provides a clean, lightweight API for running vocal pitch (F0) estimation using the RMVPE model with automatic checkpoint management.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Features

- **Inference Only**: Lightweight package focused on production inference
- **Auto-Download**: Automatic pretrained checkpoint download (~340 MB)
- **Polyphonic-Robust**: Extracts vocal pitch directly from mixed audio without source separation
- **GPU Accelerated**: Full CUDA support with automatic device detection
- **CLI Tool**: `rmvpe-infer` command for quick pitch extraction
- **Python API**: Clean programmatic interface

---

## Quick Start

### Installation

```bash
# Using pip
pip install rmvpe-infer

# Using UV (recommended)
uv pip install rmvpe-infer
```

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

## Development Installation

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

## Acknowledgments

This project builds upon the excellent work of:

- **[RMVPE](https://github.com/yxlllc/RMVPE)** by yxlllc - Original RMVPE implementation and pretrained model
- **Original Research** - Yongyi Wei et al. for the RMVPE paper

---

## License

MIT License - see [LICENSE](LICENSE) for details.

This project includes code adapted from:
- **RMVPE** by yxlllc

---

## Citation

If you use RMVPE-Infer in your research, please cite the original paper:

```bibtex
@article{wei2023rmvpe,
    title   = {RMVPE: A Robust Model for Vocal Pitch Estimation in Polyphonic Music},
    author  = {Wei, Yongyi and others},
    journal = {arXiv preprint arXiv:2306.15412},
    year    = {2023}
}
```

---

## Support

For issues and questions:
- **GitHub Issues**: [github.com/openmirlab/rmvpe-infer/issues](https://github.com/openmirlab/rmvpe-infer/issues)

---
