"""Determinism check -- must hold before any bit-exact golden fixture can be trusted.

Finding from writing this test suite: CPU inference is bit-identical
run-to-run (BatchNorm in eval mode uses fixed running stats, Dropout is a
no-op in eval mode, no other source of randomness in the forward pass), but
CUDA inference is NOT bit-identical between two runs on identical input
(observed max abs F0 diff ~1.4e-4 Hz on a real checkpoint -- ordinary
cuDNN/atomic-add nondeterminism, not a bug in this package). Consequently
the golden regression fixture (test_baseline_regression.py) is captured and
replayed on CPU specifically, and this test locks that guarantee in place.
"""

import numpy as np
import pytest

from rmvpe_infer.inference import RMVPE

from .conftest import seeded_noisy_vocal_clip


@pytest.mark.weights
def test_cpu_inference_is_bit_identical_across_runs(weights_path):
    rmvpe = RMVPE(str(weights_path), device="cpu")
    audio = seeded_noisy_vocal_clip()

    f0_a = rmvpe.infer_from_audio(audio.copy(), sample_rate=16000)
    f0_b = rmvpe.infer_from_audio(audio.copy(), sample_rate=16000)

    assert np.array_equal(f0_a, f0_b), "CPU inference should be bit-identical run-to-run"
