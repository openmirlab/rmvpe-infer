"""RMVPE — Robust Vocal Pitch Estimation (inference only).

Based on: https://github.com/yxlllc/RMVPE
Paper: "RMVPE: A Robust Model for Vocal Pitch Estimation in Polyphonic Music"

Usage:
    from rmvpe_infer import RMVPE

    model = RMVPE("rmvpe.pt")
    f0 = model.infer_from_audio(audio_array, sample_rate=16000)
"""

from .inference import RMVPE
from .download import download_model

__all__ = ["RMVPE", "download_model"]
__version__ = "0.1.0"
