"""Unit tests for the cents<->Hz decode math in utils.py -- no model, no weights.

A one-hot activation isolates the local-average window to a single bin, so
the decoded Hz must equal the closed-form cents->Hz formula exactly. This
pins down the math independent of any trained weights: if CONST or the
20-cents-per-bin spacing in constants.py ever drifts, this test catches it
without needing the real checkpoint.
"""

import numpy as np
import pytest
import torch

from rmvpe_infer.constants import CONST, N_CLASS
from rmvpe_infer.utils import to_local_average_cents, to_local_average_f0


def _cents_to_hz(cents: float) -> float:
    return 10 * 2 ** (cents / 1200)


def test_one_hot_bin_decodes_to_exact_closed_form_hz():
    bin_idx = 200
    hidden = torch.zeros(1, 1, N_CLASS)
    hidden[0, 0, bin_idx] = 1.0

    f0 = to_local_average_f0(hidden, thred=0.03)

    expected_cents = bin_idx * 20 + CONST
    expected_hz = _cents_to_hz(expected_cents)
    assert f0.shape == (1,)
    np.testing.assert_allclose(f0[0], expected_hz, rtol=1e-5)


@pytest.mark.parametrize("bin_idx", [0, 1, 180, 359])
def test_one_hot_bin_at_extremes(bin_idx):
    hidden = torch.zeros(1, 1, N_CLASS)
    hidden[0, 0, bin_idx] = 1.0
    f0 = to_local_average_f0(hidden, thred=0.03)
    expected_hz = _cents_to_hz(bin_idx * 20 + CONST)
    np.testing.assert_allclose(f0[0], expected_hz, rtol=1e-4)


def test_below_threshold_is_unvoiced():
    hidden = torch.zeros(1, 1, N_CLASS)
    hidden[0, 0, 200] = 0.01  # below default thred=0.03
    f0 = to_local_average_f0(hidden, thred=0.03)
    assert f0[0] == 0.0


def test_above_threshold_is_voiced():
    hidden = torch.zeros(1, 1, N_CLASS)
    hidden[0, 0, 200] = 0.5
    f0 = to_local_average_f0(hidden, thred=0.03)
    assert f0[0] > 0.0


def test_batched_frames_decode_independently():
    hidden = torch.zeros(1, 3, N_CLASS)
    hidden[0, 0, 100] = 1.0
    hidden[0, 1, 200] = 1.0
    hidden[0, 2, 300] = 1.0
    f0 = to_local_average_f0(hidden, thred=0.03)
    assert f0.shape == (3,)
    for i, bin_idx in enumerate([100, 200, 300]):
        expected_hz = _cents_to_hz(bin_idx * 20 + CONST)
        np.testing.assert_allclose(f0[i], expected_hz, rtol=1e-4)


def test_numpy_and_torch_decoders_agree():
    """to_local_average_cents (numpy path) and to_local_average_f0 (torch path)
    implement the same weighted-average math; cross-check them against each other."""
    rng = np.random.default_rng(0)
    salience = rng.random(N_CLASS).astype(np.float32)
    salience[200] += 5.0  # ensure a clear peak above threshold

    cents = to_local_average_cents(salience, thred=0.03)
    expected_hz = _cents_to_hz(cents)

    hidden = torch.from_numpy(salience).reshape(1, 1, N_CLASS)
    f0 = to_local_average_f0(hidden, thred=0.03)

    np.testing.assert_allclose(f0[0], expected_hz, rtol=1e-4)
