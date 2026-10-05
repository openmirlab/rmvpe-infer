# CLAUDE.md

**Distribution:** `rmvpe-infer` is not on PyPI; use the source installation in README.md.

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

### Checkpoint license (verified 2026-09-14, primary sources)

`config/checkpoints.toml`'s `license` field previously said
`"Unknown checkpoint license; verify upstream terms before redistribution"`.
Verified this round, not left as a guess — resolves to **`NOASSERTION`**
(no license grant found anywhere in the chain, matching the org convention
`scnet-infer` already uses for the same situation). Note the SPDX-precision
nuance: SPDX `NOASSERTION` technically means "no assertion made" (i.e. not
checked), while this finding is closer to SPDX `NONE` ("checked; confirmed
no license is granted"). `NOASSERTION` is used here to match the existing
sibling-package convention in this org's catalog rather than introduce a
third value; the distinction from an *unchecked* field is that this one
has verification date + sources recorded inline:

- **Code license (Dream-High/RMVPE, the paper authors' original repo)**:
  [Apache-2.0](https://github.com/Dream-High/RMVPE/blob/main/LICENSE),
  confirmed via `gh api repos/Dream-High/RMVPE --jq .license` → `apache-2.0`
  and reading the LICENSE file directly. This governs *that repo's source
  code*, not a checkpoint trained and published elsewhere.
- **Checkpoint license (yxlllc/RMVPE, a fork of Dream-High's repo, release
  `230917`)**: **none found**. `gh api repos/yxlllc/RMVPE --jq .license` →
  `null`; the fork's root listing has no `LICENSE` file at all
  (`gh api repos/yxlllc/RMVPE/contents --jq '.[].name'`); its README
  (`raw.githubusercontent.com/yxlllc/RMVPE/main/README.md`) is
  training/usage instructions only, no license statement; the `230917`
  release notes (`gh api repos/yxlllc/RMVPE/releases/tags/230917`) state
  only training-data/step-count facts, no terms.
- **Consequence**: this package's own `model.py`/`deepunet.py` (the
  architecture the shipped checkpoint loads weights into) is vendored
  near-verbatim from yxlllc/RMVPE's source (see `deepunet.py`'s module
  header) — i.e. from the *unlicensed* fork, not directly from Dream-High's
  Apache-2.0 repo. The checkpoint itself, trained by yxlllc against that
  architecture, has no license grant from any party. Per article 3's
  no-license-upstream rule, this is treated as all-rights-reserved by
  default: redistribution/commercial-use risk is real and undocumented
  until an upstream author (Dream-High or yxlllc) grants explicit terms.
  Recorded in `checkpoints.toml`'s inline comment, README's "Pretrained
  Model" + "License" sections, and `LICENSE`'s scope notice.
- **Separate, larger open question (flagged for maintainer ratification,
  not acted on this round — out of this pass's scope)**: this repo's own
  top-level `LICENSE` file has claimed "MIT License, Copyright (c) 2023
  yxlllc (original RMVPE)" since inception, but yxlllc never granted MIT
  terms for the vendored architecture code (their fork has no LICENSE at
  all). Confirmed this round: `gh api repos/Dream-High/RMVPE/contents/src
  --jq '.[].name'` lists the same filenames (`model.py`, `deepunet.py`,
  `seq.py`, `spec.py`, `utils.py`, `constants.py`) as yxlllc/RMVPE's own
  `src/` — i.e. yxlllc's fork is a modification of Dream-High's
  Apache-2.0-licensed files, not an independent rewrite. That makes the
  vendored code in this package Apache-2.0-derived, and the current
  "MIT License, Copyright (c) 2023 yxlllc" header is **very likely
  incorrect** (wrong license family and wrong attributed author) rather
  than merely an open question — but changing a repo's stated code license
  is a maintainer call this round deliberately did not make unilaterally.
  Also checked `yxlllc/RMVPE`'s issue tracker
  (`gh api 'repos/yxlllc/RMVPE/issues?state=all'`) for any license
  statement — only two unrelated ONNX-export issues exist, nothing on
  licensing. `openmirlab-dev/radar.md:1527` independently records the same
  "no LICENSE file found" finding for yxlllc/RMVPE, verified 2026-07-12 —
  consistent with, not contradicting, this round's re-verification.

## Testing

- `pytest tests/` (or `uv run pytest tests/`) runs the **CI-safe unit
  suite** (51 tests as of this writing): import smoke, model construction/
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


Push/PR CI covers all declared Python classifiers (3.10, 3.11, 3.12).
`UV_PYTHON` selects each matrix interpreter; an assertion verifies the running
version before `uv run --no-sync pytest tests/ -q`. This covers the existing
offline suite, not real-weight or network validation. Workflow permissions are
read-only. The matrix exposed an unconditional `tomllib` import on Python 3.10;
config.py now uses a conditional `tomli>=2.0` backport dependency there. Model
code, checkpoint values, and numerical dependencies are unchanged.

## Packaging

Build backend: hatchling. Version is single-sourced in
`src/rmvpe_infer/__about__.py` — don't hand-edit a version literal in
`pyproject.toml` or `__init__.py` directly; `pyproject.toml` reads it via
`[tool.hatch.version]`.

## Dev workflow

Uses `uv`. `uv sync --extra dev` gets pytest. Use `uv run`, not bare
`python`, so the editable install and dependency versions match CI.

## Distribution policy (2026-10-05)

Install the current source from `https://github.com/openmirlab/rmvpe-infer`. GitHub release workflows verify and build distributions but do not upload to PyPI. Existing PyPI versions, where any exist, are historical snapshots. Update installation examples to use Git when changing this package.
