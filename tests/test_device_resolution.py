"""Device-resolution unit tests for RMVPE.__init__ -- no real checkpoint required.

`E2E0` (the actual neural net) and `torch.load` are stubbed out so the
device-resolution branch at the top of `RMVPE.__init__` can be exercised
directly against the real constructor without needing a cached checkpoint.
Everything else (MelSpectrogram, torch.device) runs for real.

Regression coverage: a literal `"auto"` string used to fall through to
`torch.device("auto")` and raise, since only `None` was treated as
auto-detect. It must now resolve identically to the unset (`None`) default.
"""

import pytest
import torch

from rmvpe_infer import inference as inference_module
from rmvpe_infer.inference import RMVPE


class _StubModel:
    """Stands in for E2E0: accepts load_state_dict/eval/to without real weights."""

    def load_state_dict(self, state_dict):
        pass

    def eval(self):
        return self

    def to(self, device):
        self.device = device
        return self


class _StubMel:
    def __init__(self, *args, **kwargs):
        pass

    def to(self, device):
        self.device = device
        return self


@pytest.fixture(autouse=True)
def _stub_checkpoint_loading(monkeypatch):
    monkeypatch.setattr(inference_module, "E2E0", lambda *args, **kwargs: _StubModel())
    monkeypatch.setattr(inference_module, "MelSpectrogram", _StubMel)
    monkeypatch.setattr(inference_module.torch, "load", lambda *args, **kwargs: {"model": {}})


def _expected_auto_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def test_device_none_resolves_to_auto_detected_device():
    rmvpe = RMVPE("fake.pt")
    assert str(rmvpe.device) == _expected_auto_device()


def test_device_auto_string_resolves_the_same_as_unset():
    """The literal "auto" sentinel must not crash and must match the None default."""
    rmvpe = RMVPE("fake.pt", device="auto")
    assert str(rmvpe.device) == _expected_auto_device()


def test_device_none_and_auto_produce_identical_devices():
    default = RMVPE("fake.pt", device=None)
    auto = RMVPE("fake.pt", device="auto")
    assert str(default.device) == str(auto.device)


def test_explicit_device_is_still_respected():
    rmvpe = RMVPE("fake.pt", device="cpu")
    assert str(rmvpe.device) == "cpu"


def test_explicit_cuda_index_is_validated_and_reaches_model(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    rmvpe = RMVPE("fake.pt", device="cuda:1")
    assert rmvpe.device == torch.device("cuda:1")
    assert rmvpe.model.device == torch.device("cuda:1")


def test_unavailable_or_invalid_explicit_devices_raise(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        RMVPE("fake.pt", device="cuda")
    with pytest.raises(RuntimeError, match="MPS"):
        RMVPE("fake.pt", device="mps")
    with pytest.raises(ValueError):
        RMVPE("fake.pt", device="metal")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(RuntimeError, match="index 1"):
        RMVPE("fake.pt", device="cuda:1")
