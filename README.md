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

The pretrained checkpoint is automatically downloaded from the [official RMVPE release](https://github.com/yxlllc/RMVPE/releases/tag/230917) (~340 MB).

- **Architecture**: Deep U-Net encoder-decoder + BiGRU
- **Training data**: MIR-1K, PTDB, M4Singer
- **Frame rate**: 10ms (100 fps at 16kHz)
- **Frequency range**: ~30 Hz to 8000 Hz
- **Cache location**: `~/.cache/rmvpe/`

---

## Development Installation

```bash
# Clone repository
git clone https://github.com/openmirlab/rmvpe-infer.git
cd rmvpe-infer

# Install with UV
uv sync

# Install with pip
pip install -e ".[dev]"
```

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
