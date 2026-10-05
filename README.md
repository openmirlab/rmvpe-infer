# rmvpe-infer

Estimate vocal pitch (F0) from an audio file or a mono audio array using
[RMVPE](https://github.com/yxlllc/RMVPE). This package provides inference,
checkpoint download, and a command-line tool; it does not train a model or
separate vocals. The repository is public, but **`rmvpe-infer` is not currently
published on PyPI**.

## Why this exists

The [original RMVPE research](https://github.com/Dream-High/RMVPE) estimates
vocal pitch directly from polyphonic music. This repository packages its
inference path behind a small Python API and CLI, with checkpoint caching.

## Acknowledgments

- **Haojie Wei, Xueke Cao, Tangpeng Dan, and Yueguo Chen** developed RMVPE
  and published the [INTERSPEECH 2023 paper](https://doi.org/10.21437/Interspeech.2023-528).
- **[Dream-High/RMVPE](https://github.com/Dream-High/RMVPE)** is the
  original research code repository.
- **[yxlllc/RMVPE](https://github.com/yxlllc/RMVPE)** is the fork from
  which this package adapts inference code and whose
  [release 230917](https://github.com/yxlllc/RMVPE/releases/tag/230917)
  hosts the default checkpoint.

## Citation

Please cite the original paper when using this model:

```bibtex
@inproceedings{wei23b_interspeech,
  title     = {{RMVPE: A Robust Model for Vocal Pitch Estimation in Polyphonic Music}},
  author    = {Haojie Wei and Xueke Cao and Tangpeng Dan and Yueguo Chen},
  year      = {2023},
  booktitle = {{Interspeech 2023}},
  pages     = {5421--5425},
  doi       = {10.21437/Interspeech.2023-528},
  issn      = {2958-1796},
}
```

## Scope

The public API is `RMVPE`, `RMVPESession`, and `download_model`.
Training, other pitch estimators, source separation, and voice conversion
are outside this inference-only package.

## Install

Python 3.10 or newer is required. Install the current source with:

```bash
python -m pip install "git+https://github.com/openmirlab/rmvpe-infer.git"
```

For a reproducible application, replace the Git URL's default branch with a
reviewed commit, for example
`git+https://github.com/openmirlab/rmvpe-infer.git@014572b7d4fdafb570396da4fb350eb627e6e279`.
The install includes PyTorch, torchaudio, NumPy, and librosa; choose a PyTorch
build appropriate for your device if you need CUDA.

To work on this repository:

```bash
git clone https://github.com/openmirlab/rmvpe-infer.git
cd rmvpe-infer
uv sync --extra dev
```

## Run pitch estimation

The first inference downloads the upstream checkpoint unless it is already
cached or you supply one. The command writes CSV columns `timestamp_s` and
`f0_hz`; zero Hz denotes an unvoiced frame.

```bash
rmvpe-infer -i vocals.wav -o f0.csv
rmvpe-infer -i vocals.wav -o f0.csv --threshold 0.05 --viterbi --device cuda
```

For a mono NumPy array, use the reusable session when processing multiple
buffers. The context manager loads the model once and releases it on exit.

```python
import librosa
from rmvpe_infer import RMVPESession

audio, sample_rate = librosa.load("vocals.wav", sr=None, mono=True)
with RMVPESession() as session:
    f0_hz = session.infer(audio, sample_rate=sample_rate)
```

If you manage the model yourself, `RMVPE(str(download_model()))` loads the
default checkpoint. Its `infer_from_audio(audio, sample_rate=...)` method
returns the same one-dimensional NumPy array of pitch values. Both APIs
resample input to the model's 16 kHz rate.

| Setting | Where to set it | Default | Effect |
| --- | --- | --- | --- |
| `device` | `RMVPESession(...)`, `RMVPE(...)`, CLI `--device` | CUDA if available, otherwise CPU | Accepts `cpu`, `cuda`, `cuda:N`, or `auto`; MPS is unsupported. |
| `hop_length` | constructors, CLI `--hop-length` | 160 | Samples per output frame at 16 kHz (10 ms). |
| `thred` | `infer(...)`, `infer_from_audio(...)`; CLI `--threshold` | 0.03 | Voicing threshold. The Python spelling is `thred`. |
| `use_viterbi` | `infer(...)`, `infer_from_audio(...)`; CLI `--viterbi` | `False` | Enables Viterbi pitch decoding. |

The `device` argument accepted by `infer_from_audio` is retained for
compatibility but ignored; set the device when constructing the model or
session.

## Checkpoint and offline use

The default checkpoint comes from [yxlllc/RMVPE release 230917](https://github.com/yxlllc/RMVPE/releases/tag/230917),
not from this repository or an OpenMIRLab mirror. On first use,
`download_model()` extracts it to `~/.cache/rmvpe/`. It verifies SHA-256
`19dc1809cf4cdb0a18db93441816bc327e14e5644b72eeaae5220560c6736fe2`
on download and on subsequent default-cache use. A mismatch raises
`ChecksumMismatchError`.

To use a checkpoint you already have, set `RMVPE_INFER_WEIGHTS` to its `.pt`
path, pass `model_path` to `RMVPESession`, or use the CLI's `--model` option.
These explicit local paths **bypass the built-in checksum check**; verify
their provenance yourself. The package and its Git history do not include
the checkpoint.

## License and upstream rights

This repository currently declares MIT in its [LICENSE](LICENSE) and package
metadata. That declaration does **not** grant rights to the separately
downloaded checkpoint. We found no explicit checkpoint license in the
[upstream fork](https://github.com/yxlllc/RMVPE) or its
[release notes](https://github.com/yxlllc/RMVPE/releases/tag/230917); the
checkpoint catalog therefore records `NOASSERTION`. Also, the provenance and
license attribution of the adapted architecture code require review: the
[original repository declares Apache-2.0](https://github.com/Dream-High/RMVPE/blob/main/LICENSE),
while the fork has no apparent license file. Do not assume this repository's
MIT label resolves those upstream rights. Obtain clarification from the
upstream authors before relying on redistribution or commercial-use rights
for the checkpoint or adapted code.

## Development

The default checkpoint URL, checksum, and license observation live in
[`src/rmvpe_infer/config/checkpoints.toml`](src/rmvpe_infer/config/checkpoints.toml).

```bash
uv run pytest tests/       # offline tests; no checkpoint required
uv run pytest tests/ -m weights  # opt-in tests using a real checkpoint
uv run ruff check .
```

The weights tests may download the checkpoint when no local copy is set.
For package maintenance details, see [CLAUDE.md](CLAUDE.md). Report issues
through [GitHub Issues](https://github.com/openmirlab/rmvpe-infer/issues).
