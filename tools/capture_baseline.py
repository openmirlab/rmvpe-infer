"""Regenerate tests/fixtures/baseline_f0.json -- the golden regression fixture.

Run this after any change that could alter RMVPE's numerical output (model
code, spec.py's mel extraction, utils.py's decode math, or a checkpoint
swap). It always runs on CPU so the fixture is reproducible across machines
(see tests/test_determinism.py: CUDA inference on this model is not
bit-identical run-to-run, CPU is). Requires the real checkpoint --
set RMVPE_INFER_WEIGHTS or have it cached at ~/.cache/rmvpe/.

Usage:
    uv run python tools/capture_baseline.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "tests"))

from conftest import seeded_noisy_vocal_clip  # noqa: E402

from rmvpe_infer.download import download_model  # noqa: E402
from rmvpe_infer.inference import RMVPE  # noqa: E402

FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "baseline_f0.json"
SEED = 1234
SAMPLE_RATE = 16000


def build_fixture():
    model_path = download_model()
    rmvpe = RMVPE(str(model_path), device="cpu")

    audio = seeded_noisy_vocal_clip(sr=SAMPLE_RATE, duration=1.0, seed=SEED)
    f0 = rmvpe.infer_from_audio(audio, sample_rate=SAMPLE_RATE)

    return {
        "meta": {
            "torch_version": torch.__version__,
            "device": "cpu",
            "seed": SEED,
            "sample_rate": SAMPLE_RATE,
            "hop_length": rmvpe.hop_length,
            "captured_at": datetime.now(timezone.utc).isoformat(),
        },
        # Full double precision (no rounding): a float32 value upcast to
        # Python float and back down to float32 round-trips exactly, which
        # is what test_baseline_regression.py relies on for bit-exactness.
        "f0": [float(x) for x in f0],
    }


def main():
    fixture = build_fixture()
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FIXTURE_PATH, "w") as f:
        json.dump(fixture, f, indent=2)
    print(f"Wrote {FIXTURE_PATH} ({len(fixture['f0'])} frames)")


if __name__ == "__main__":
    main()
