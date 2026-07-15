# Changelog

All notable changes to rmvpe-infer will be documented in this file.

## Unreleased

### Added
- `RMVPESession` explicit lifecycle facade with ready-only inference,
  `load`, `release`, `close`, status, cache metadata, and context-manager
  support while preserving the existing `RMVPE` API.
- Package-owned `config/checkpoints.toml` with checksum/provenance metadata and
  generic URL/checksum/config overrides.

## [0.1.0] - 2026-07-12

### Added
- Test suite (previously zero tests): 31 CI-safe unit tests (import smoke,
  model construction/forward-pass shape+range sanity, decode-math unit
  tests, mel spectrogram sanity, download.py checksum/env-var/cache logic
  with network mocked out) plus 6 tests requiring the real checkpoint
  (`pytest -m weights`): a physics accuracy test synthesizing 220/440/880Hz
  tones and a vibrato sweep with mathematical ground truth, a determinism
  check, and a golden regression fixture (`tests/fixtures/baseline_f0.json`,
  captured/replayed on CPU — see CLAUDE.md for why CUDA inference isn't
  bit-reproducible run-to-run on this model).
- `RMVPE_INFER_WEIGHTS` environment variable: point at an existing
  checkpoint to skip the network/cache entirely (used by tests, useful for
  offline/air-gapped use).
- sha256 verification of the downloaded/cached checkpoint
  (`download.verify_checksum`, `download.MODEL_SHA256`) — a corrupted,
  truncated, or tampered checkpoint now raises `ChecksumMismatchError`
  instead of silently loading.
- `.github/workflows/test.yml`: CI now runs the unit suite on every push/PR
  (previously only `publish.yml` existed, with no test gate at all).
- Package version single-sourced in `src/rmvpe_infer/__about__.py`
  (`pyproject.toml`'s `[tool.hatch.version]` reads it; `__init__.py`
  imports it — no more hand-maintained duplicate literals).
- nav headers (title + rationale + `Reads:` line) on every module.

### Changed
- `publish.yml`'s `publish` job now `needs: [test]` — publishing gates on
  the test suite passing, per the org constitution's release rule.

### Known gap (tracked, not fixed this round)
- The pretrained checkpoint downloads from yxlllc's own GitHub release, a
  third-party host — not yet mirrored under openmirlab control. Flagged in
  CLAUDE.md/README for follow-up; re-hosting is an outward action requiring
  explicit sign-off per the org constitution.
