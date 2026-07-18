"""CLI regression tests -- `--device` must actually reach RMVPE()'s constructor.

Before the fix, `cli.main()` parsed `--device` but only forwarded it to
`infer_from_audio(..., device=...)`, whose `device` parameter is documented
in inference.py as ignored ("uses device set at init"). RMVPE() itself was
always constructed with its own unset default, so `--device cuda:0` was a
silent no-op. These tests invoke the real `cli.main()` entrypoint (via
`sys.argv`, exactly as a user would run the command) with model loading and
audio decoding mocked out, then inspect the *resolved* `.device` on the
`RMVPE` instance the CLI actually constructed.

Reads: rmvpe_infer.cli (main), rmvpe_infer.inference (RMVPE, patched here).
"""

import sys

import numpy as np
import pytest

from rmvpe_infer import cli
from rmvpe_infer import inference as inference_module


class _RecordingRuntime:
    """Stands in for RMVPE: records the device it was constructed with and
    mirrors RMVPE.__init__'s own None/"auto" -> "cpu" auto-detect resolution
    (this test suite runs without a GPU), so assertions reflect the same
    "resolved device" a real construction would produce.
    """

    last_instance = None

    def __init__(self, model_path, hop_length=160, device=None):
        self.model_path = model_path
        self.hop_length = hop_length
        self.device = "cpu" if device in (None, "auto") else device
        _RecordingRuntime.last_instance = self

    def infer_from_audio(self, audio, sample_rate=16000, thred=0.03, use_viterbi=False, **kwargs):
        return np.zeros(4, dtype=np.float32)


@pytest.fixture(autouse=True)
def _reset_recorder():
    _RecordingRuntime.last_instance = None
    yield
    _RecordingRuntime.last_instance = None


def _run_cli(monkeypatch, tmp_path, extra_args):
    monkeypatch.setattr(inference_module, "RMVPE", _RecordingRuntime)

    def fake_load(path, sr=None, mono=True):
        return np.zeros(16000, dtype=np.float32), 16000

    monkeypatch.setattr("librosa.load", fake_load)

    output = tmp_path / "f0.csv"
    argv = ["rmvpe-infer", "-i", "fake.wav", "-o", str(output), "-m", "fake_model.pt", *extra_args]
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    return output


def test_cli_device_flag_reaches_constructed_rmvpe_instance(monkeypatch, tmp_path):
    """`--device cuda:0` must reach RMVPE()'s constructor, not just infer_from_audio()."""
    _run_cli(monkeypatch, tmp_path, ["--device", "cuda:0"])
    assert _RecordingRuntime.last_instance is not None
    assert _RecordingRuntime.last_instance.device == "cuda:0"


def test_cli_device_omitted_falls_back_to_auto_detect(monkeypatch, tmp_path):
    _run_cli(monkeypatch, tmp_path, [])
    assert _RecordingRuntime.last_instance.device == "cpu"


def test_cli_device_auto_is_accepted_explicitly(monkeypatch, tmp_path):
    _run_cli(monkeypatch, tmp_path, ["--device", "auto"])
    assert _RecordingRuntime.last_instance.device == "cpu"
