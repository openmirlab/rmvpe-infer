"""Golden regression: seeded synthetic clip's F0 curve must match the
recorded fixture bit-exactly on CPU.

Per the org constitution's amendment on float-digest fixtures (bit-exact
assertions are only valid on the recording torch-build/hardware): this
fixture was captured on CPU (see tools/capture_baseline.py and
test_determinism.py -- CPU inference on this model is verified
bit-identical run-to-run; CUDA is not). The test always re-runs on CPU
too, so the only thing that can invalidate it across machines is a
different torch version, which is checked explicitly and SKIPS with a
reason rather than failing.

Regenerate with: `uv run python tools/capture_baseline.py`
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from rmvpe_infer.inference import RMVPE

from .conftest import seeded_noisy_vocal_clip

REPO_ROOT = Path(__file__).parent.parent
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "baseline_f0.json"


@pytest.mark.weights
def test_baseline_f0_matches_fixture_on_recording_torch_version(weights_path):
    with open(FIXTURE_PATH) as f:
        fixture = json.load(f)

    meta = fixture["meta"]
    if torch.__version__ != meta["torch_version"]:
        pytest.skip(
            f"fixture recorded on torch {meta['torch_version']}, "
            f"running {torch.__version__}: bit-exact float digests are only "
            f"valid on the recording torch build (org constitution art. 2)"
        )

    rmvpe = RMVPE(str(weights_path), device="cpu")
    audio = seeded_noisy_vocal_clip(
        sr=meta["sample_rate"], duration=1.0, seed=meta["seed"]
    )
    actual_f0 = rmvpe.infer_from_audio(audio, sample_rate=meta["sample_rate"])

    expected_f0 = np.array(fixture["f0"], dtype=np.float32)
    assert actual_f0.shape == expected_f0.shape
    assert np.array_equal(actual_f0, expected_f0), (
        "F0 curve diverged from the recorded golden fixture -- this is an "
        "accuracy regression, not a rounding difference (see determinism "
        "test: CPU inference is bit-identical run-to-run)"
    )
