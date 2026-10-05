# Changelog

## Unreleased — distribution policy

- Stop publishing new versions to PyPI; GitHub source is the maintained installation channel. GitHub release CI continues to run verification and build checks.

All notable changes to rmvpe-infer will be documented in this file.

## Unreleased


- Correct README installation guidance for the current Git-only distribution of `rmvpe-infer`.
### Fixed
- Restore Python 3.10 package imports with a declared conditional `tomli` backport
  for checkpoint configuration; newer Python retains stdlib `tomllib`.

### CI
- Add the missing Python 3.11 offline CI job and verify each selected interpreter,
  matching the existing 3.10–3.12 support classifiers.

### Removed
- MPS device support (org decision 2026-09-14). Apple MLX/MPS backends are
  permanently out of scope (org canon `openmirlab-dev` 5e588e6, art. 4b).
  `device="mps"` now raises `ValueError` unconditionally instead of
  resolving; `"auto"` never selects MPS. Supported device vocabulary:
  `"auto"`, `"cpu"`, `"cuda"`, `"cuda:N"` (plus `None`, treated as `"auto"`).
  MPS support was never part of a released version — it was added and
  removed within this same `Unreleased` section — so this is not a breaking
  change for any published release.

### Changed
- `config/checkpoints.toml`'s checkpoint `license` field: verified against
  primary sources (Dream-High/RMVPE's Apache-2.0 code license vs. the
  unlicensed yxlllc/RMVPE fork that actually trained and released the
  checkpoint) and set to the honest, verified value `"NOASSERTION"`,
  replacing the previous unverified `"Unknown checkpoint license..."`
  placeholder. README/CLAUDE.md/LICENSE updated to document the finding and
  its redistribution/commercial-use consequence; see CLAUDE.md's "Checkpoint
  license" section for the full sourced writeup.

### Added
- Test asserting the checkpoint catalog's `license` field is a real,
  non-empty, non-placeholder value (`tests/test_config.py`).
- `RMVPESession` explicit lifecycle facade with ready-only inference,
  `load`, `release`, `close`, status, cache metadata, and context-manager
  support while preserving the existing `RMVPE` API.
- Package-owned `config/checkpoints.toml` with checksum/provenance metadata and
  generic URL/checksum/config overrides.
- Strict device validation for `cpu`, `cuda`, and `cuda:N`, while preserving
  legacy automatic CUDA-or-CPU selection. (Originally also validated `mps`;
  removed later in this same `Unreleased` section, see "Removed" above.)

### Changed
- `RMVPESession.release()` is reloadable and `close()` is terminal/idempotent;
  cache inspection now shares the loader resolver without downloading or
  materializing directories.
- Public download URL/hash constants are now derived from packaged checkpoint
  TOML metadata; custom session overrides affect both load and cache reporting.

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
