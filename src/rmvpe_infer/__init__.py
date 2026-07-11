"""RMVPE — Robust Vocal Pitch Estimation (inference only).

Based on: https://github.com/yxlllc/RMVPE
Paper: "RMVPE: A Robust Model for Vocal Pitch Estimation in Polyphonic Music"
Public surface: `RMVPE` (inference.py) for pitch estimation, `download_model`
(download.py) for checkpoint retrieval. `__version__` is single-sourced from
`__about__.py` — never hand-edit a version literal here or in pyproject.toml.

Usage:
    from rmvpe_infer import RMVPE

    model = RMVPE("rmvpe.pt")
    f0 = model.infer_from_audio(audio_array, sample_rate=16000)

Reads: __about__.py, inference.py, download.py.
"""

from .__about__ import __version__
from .inference import RMVPE
from .download import download_model

__all__ = ["RMVPE", "download_model", "__version__"]
