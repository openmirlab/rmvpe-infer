"""MelSpectrogram feature-extraction sanity -- no weights required.

Checks the shape/dtype contract inference.py relies on: `MelSpectrogram`
must turn (B, T_samples) audio into (B, N_MELS, T_frames) log-mel features
with no NaNs, using the exact constants (N_MELS/SAMPLE_RATE/WINDOW_LENGTH/
MEL_FMIN/MEL_FMAX) the real checkpoint was trained against.
"""

import torch

from rmvpe_infer.constants import MEL_FMAX, MEL_FMIN, N_MELS, SAMPLE_RATE, WINDOW_LENGTH
from rmvpe_infer.spec import MelSpectrogram

HOP_LENGTH = 160


def _extractor():
    return MelSpectrogram(N_MELS, SAMPLE_RATE, WINDOW_LENGTH, HOP_LENGTH, None, MEL_FMIN, MEL_FMAX)


def test_output_shape_and_dtype():
    mel_extractor = _extractor()
    audio = torch.randn(1, SAMPLE_RATE)  # 1 second at 16kHz
    mel = mel_extractor(audio, center=True)

    assert mel.dim() == 3
    assert mel.shape[0] == 1
    assert mel.shape[1] == N_MELS
    assert mel.dtype == torch.float32
    assert not torch.isnan(mel).any()


def test_more_samples_yields_more_frames():
    mel_extractor = _extractor()
    short = mel_extractor(torch.randn(1, SAMPLE_RATE), center=True)
    long = mel_extractor(torch.randn(1, 2 * SAMPLE_RATE), center=True)
    assert long.shape[-1] > short.shape[-1]


def test_clamp_prevents_log_of_zero():
    """Silence (all zeros) must not produce -inf/NaN thanks to the clamp floor."""
    mel_extractor = _extractor()
    silence = torch.zeros(1, SAMPLE_RATE)
    mel = mel_extractor(silence, center=True)
    assert torch.isfinite(mel).all()
