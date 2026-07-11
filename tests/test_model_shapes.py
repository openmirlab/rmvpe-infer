"""Model construction/forward-pass sanity -- no checkpoint required.

Builds E2E0 (the architecture inference.py's RMVPE actually loads weights
into: `E2E0(4, 1, (2, 2))`) with random init and checks structural
invariants that would catch an accidental architecture change even before
any accuracy test runs: parameter count, output shape, dtype, and the
[0, 1] range the final Sigmoid guarantees.

The encoder/decoder halves the time axis 5 times (once per en/de layer), so
forward() requires the input frame count to be a multiple of 32 -- this is
why inference.py's `mel2hidden` reflect-pads to the next multiple of 32
before calling the model; passing an unaligned length raises, which we
assert explicitly so that constraint doesn't silently drift.

Reads: rmvpe_infer.model (E2E0), rmvpe_infer.constants (N_MELS, N_CLASS).
"""

import pytest
import torch

from rmvpe_infer.constants import N_CLASS, N_MELS
from rmvpe_infer.model import E2E0

# The exact constructor args inference.py's RMVPE uses. If this drifts,
# loading the real checkpoint's state_dict would fail with a shape
# mismatch -- pinning it here catches that at unit-test speed.
E2E0_ARGS = (4, 1, (2, 2))

# Regression pin: this is E2E0(4, 1, (2, 2))'s exact parameter count. An
# unexplained change here means the architecture changed and the shipped
# checkpoint will (silently or loudly) stop matching.
EXPECTED_PARAM_COUNT = 91_996_477


@pytest.fixture
def model():
    torch.manual_seed(0)
    m = E2E0(*E2E0_ARGS)
    m.eval()
    return m


def test_param_count_matches_shipped_architecture(model):
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == EXPECTED_PARAM_COUNT


@pytest.mark.parametrize("batch, frames", [(1, 32), (1, 64), (2, 32)])
def test_forward_pass_output_shape_dtype_range(model, batch, frames):
    mel = torch.randn(batch, N_MELS, frames)
    with torch.no_grad():
        out = model(mel)

    assert out.shape == (batch, frames, N_CLASS)
    assert out.dtype == torch.float32
    # Final layer is nn.Sigmoid(), so every value must be in [0, 1].
    assert torch.all(out >= 0.0)
    assert torch.all(out <= 1.0)
    assert not torch.isnan(out).any()


def test_forward_requires_frame_count_multiple_of_32(model):
    """Unaligned T raises -- this is exactly why mel2hidden pads before calling
    the model. If this stops raising, an implicit assumption elsewhere broke."""
    mel = torch.randn(1, N_MELS, 50)  # not a multiple of 32
    with pytest.raises(RuntimeError):
        with torch.no_grad():
            model(mel)


def test_eval_mode_is_deterministic(model):
    """Dropout(0.25) sits in the final fc Sequential; in eval() mode it must be
    a no-op so two forward passes on identical input are bit-identical."""
    mel = torch.randn(1, N_MELS, 64)
    with torch.no_grad():
        out1 = model(mel)
        out2 = model(mel)
    assert torch.equal(out1, out2)
