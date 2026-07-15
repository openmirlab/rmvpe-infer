# CLAUDE.md

Guidance for Claude Code (or any agent) working in this repository. Follows
the [openmirlab org constitution](https://github.com/openmirlab/openmirlab-skills/blob/main/plugins/openmirlab/CLAUDE.md).

## What this is

`rmvpe_infer` is an inference-only, single-model wrapper around
[yxlllc/RMVPE](https://github.com/yxlllc/RMVPE) for vocal pitch (F0)
estimation. Public surface: `RMVPE` (`inference.py`), `RMVPESession`
(`session.py`), and `download_model` (`download.py`), all re-exported from
`rmvpe_infer/__init__.py`.

**Scope / status:** shipped and stable. There is no training path, no other
pitch models, and no downstream consumer logic (source separation, voice
conversion) in this repo by design — see README's Scope section for the
"out of scope, forever" list. The one open item is weight hosting (see
below): functionally done, org-mirror migration still pending.

```
src/rmvpe_infer/
  __about__.py      -- single-sourced __version__ (read by pyproject.toml)
  __init__.py       -- public exports: RMVPE, download_model, __version__
  inference.py      -- RMVPE class: load checkpoint, resample, decode F0
  model.py          -- E2E0/E2E: DeepUnet + conv + BiGRU + sigmoid heads
  deepunet.py        -- residual conv encoder/decoder backbone
  seq.py             -- BiGRU/BiLSTM recurrent heads
  spec.py            -- MelSpectrogram feature extraction
  utils.py           -- cents<->Hz decode (local-average, Viterbi)
  constants.py        -- fixed model/audio constants (must match training)
  download.py         -- checkpoint download, sha256 verification, cache
  config.py           -- package-owned checkpoint metadata and overrides
  config/checkpoints.toml -- release-pinned URL, checksum, provenance
  session.py          -- explicit load/infer/release/close lifecycle
  cli.py              -- `rmvpe-infer` command
tests/                -- pytest, see "Testing" below
tools/capture_baseline.py -- regenerates the golden regression fixture
```

## Model weights (org constitution article 4)

The checkpoint auto-downloads from the official yxlllc/RMVPE GitHub release
(`https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip`, ~350MB
extracted `.pt`) to `~/.cache/rmvpe/`. **This is a third-party host, not yet
an openmirlab mirror** — the constitution wants weights hosted under org
control (HF account or mirror as fallback). This is a known gap flagged for
follow-up (outward action, needs human sign-off per article 8); this round
only added sha256 verification (`download.MODEL_SHA256`,
`19dc1809cf4cdb0a18db93441816bc327e14e5644b72eeaae5220560c6736fe2`) and an
`RMVPE_INFER_WEIGHTS` env var override so tests/CI/offline users can point
at a local copy without touching the network or the third-party URL at all.

## Testing

- `pytest tests/` (or `uv run pytest tests/`) runs the **CI-safe unit
  suite** (31 tests as of this writing): import smoke, model construction/
  forward-pass shape+range sanity (no checkpoint needed, random init),
  decode-math unit tests, mel spectrogram sanity, and download.py's
  checksum/env-var/cache logic (network mocked out). No weights, no
  network, no GPU required.
- Two pytest markers gate everything else, both excluded by default via
  `pyproject.toml`'s `addopts`:
  - `weights` — needs the real checkpoint (`RMVPE_INFER_WEIGHTS` or
    `~/.cache/rmvpe/`); run with `pytest -m weights`. Covers:
    - `test_pitch_physics.py` — synthesizes 220/440/880Hz tones + a vibrato
      sweep, asserts detected F0 is within a few cents of the mathematical
      ground truth in voiced regions. This is the accuracy centerpiece —
      real ground truth, not a recorded fixture.
    - `test_determinism.py` — CPU inference is bit-identical run-to-run;
      **CUDA is not** (observed ~1.4e-4 Hz max diff between two runs on
      identical input — ordinary cuDNN/atomic-add nondeterminism). This is
      why the golden fixture below is captured and replayed on CPU only.
    - `test_baseline_regression.py` — a seeded synthetic clip's F0 curve
      must match `tests/fixtures/baseline_f0.json` bit-exactly on CPU.
      Skips (doesn't fail) if the running torch version differs from the
      one recorded in the fixture's `meta` (org constitution: bit-exact
      float digests are only valid on the recording torch build).
  - `network` — reserved for future URL-liveness checks; none registered
    yet (the one external URL is the weights download itself, exercised by
    the `weights`-marked tests above once cached).
- Regenerate the golden fixture after any change to model/spec/utils code
  or a checkpoint swap: `uv run python tools/capture_baseline.py`.
- CI: `.github/workflows/test.yml` runs the CI-safe suite on push/PR;
  `.github/workflows/publish.yml`'s `publish` job `needs: [test]`.

## Packaging

Build backend: hatchling. Version is single-sourced in
`src/rmvpe_infer/__about__.py` — don't hand-edit a version literal in
`pyproject.toml` or `__init__.py` directly; `pyproject.toml` reads it via
`[tool.hatch.version]`.

## Dev workflow

Uses `uv`. `uv sync --extra dev` gets pytest. Use `uv run`, not bare
`python`, so the editable install and dependency versions match CI.
