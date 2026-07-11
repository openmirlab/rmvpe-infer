"""Physics accuracy test: synthesize known-frequency tones, assert RMVPE
recovers them within a few cents. This is a REAL ground-truth accuracy
test (mathematical, not a recorded fixture) -- stronger evidence than a
golden regression that the model is doing its job at all.

Requires the real checkpoint (`weights` marker; skips cleanly via the
`weights_path` fixture in conftest.py if none is cached locally --
RMVPE_INFER_WEIGHTS or ~/.cache/rmvpe/). Runs on CPU explicitly for
reproducible results across machines (see test_determinism.py for why:
CUDA inference on this model is not bit-reproducible run-to-run).

Tolerance notes (measured empirically while writing this test, see the
docstrings on each test): pure sine tones lack the harmonic structure
RMVPE was trained on (real singing voice), so voicing confidence dips
below the default 0.03 threshold on some frames especially at the higher
880Hz tone -- this is an out-of-distribution-input quirk, not a bug. Tests
therefore only assert accuracy IN voiced regions, plus a floor on the
voiced fraction to catch a wholesale detection failure.
"""

import numpy as np
import pytest

from rmvpe_infer.inference import RMVPE

from .conftest import sine_tone, vibrato_tone

# Edge frames (first/last few hops) can carry STFT/reflect-padding boundary
# artifacts unrelated to the model's steady-state pitch tracking accuracy;
# trim them before computing accuracy statistics.
EDGE_TRIM = 5

# A single octave-slip or double/half-pitch error would show as ~1200 cents;
# genuine tracking error for a clean tone is a few cents. 25 cents is a
# generous ceiling that still catches any real accuracy regression.
MAX_MEDIAN_CENTS_ERROR = 25.0
MIN_VOICED_FRACTION = 0.5


def _cents_error(detected_hz, ground_truth_hz):
    return 1200 * np.log2(detected_hz / ground_truth_hz)


@pytest.fixture(scope="module")
def rmvpe(weights_path):
    return RMVPE(str(weights_path), device="cpu")


@pytest.mark.weights
@pytest.mark.parametrize("freq_hz", [220.0, 440.0, 880.0])
def test_pure_tone_pitch_within_a_few_cents(rmvpe, freq_hz):
    audio = sine_tone(freq_hz)
    f0 = rmvpe.infer_from_audio(audio, sample_rate=16000)

    trimmed = f0[EDGE_TRIM:-EDGE_TRIM]
    voiced = trimmed[trimmed > 0]
    voiced_fraction = len(voiced) / len(trimmed)

    assert voiced_fraction >= MIN_VOICED_FRACTION, (
        f"{freq_hz} Hz: only {voiced_fraction:.0%} of frames were voiced"
    )

    median_hz = float(np.median(voiced))
    err_cents = _cents_error(median_hz, freq_hz)
    assert abs(err_cents) <= MAX_MEDIAN_CENTS_ERROR, (
        f"{freq_hz} Hz: median detected {median_hz:.2f} Hz "
        f"({err_cents:+.1f} cents), exceeds {MAX_MEDIAN_CENTS_ERROR} cent tolerance"
    )


@pytest.mark.weights
def test_vibrato_sweep_tracks_instantaneous_frequency(rmvpe):
    """A vibrato sweep is a stronger test than a static tone: the ground
    truth changes every sample, so this checks RMVPE is actually tracking
    pitch over time rather than returning a lucky constant."""
    carrier, depth, rate = 440.0, 20.0, 5.0
    audio, inst_freq = vibrato_tone(carrier=carrier, depth=depth, rate=rate)
    f0 = rmvpe.infer_from_audio(audio, sample_rate=16000)

    hop_length = 160
    frame_times_s = np.arange(len(f0)) * hop_length / 16000
    ground_truth = carrier + depth * np.sin(2 * np.pi * rate * frame_times_s)

    trimmed_f0 = f0[EDGE_TRIM:-EDGE_TRIM]
    trimmed_gt = ground_truth[EDGE_TRIM:-EDGE_TRIM]
    voiced_mask = trimmed_f0 > 0

    voiced_fraction = voiced_mask.mean()
    assert voiced_fraction >= 0.9, f"vibrato: only {voiced_fraction:.0%} voiced"

    err_cents = _cents_error(trimmed_f0[voiced_mask], trimmed_gt[voiced_mask])
    mean_abs_err = float(np.mean(np.abs(err_cents)))
    max_abs_err = float(np.max(np.abs(err_cents)))

    assert mean_abs_err <= 15.0, f"vibrato: mean |cents error| {mean_abs_err:.2f} too high"
    assert max_abs_err <= 50.0, f"vibrato: worst-frame |cents error| {max_abs_err:.2f} too high"
