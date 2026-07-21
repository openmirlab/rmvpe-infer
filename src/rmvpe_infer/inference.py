"""RMVPE: the public inference class — load a checkpoint, run pitch estimation.

Owns the full pipeline: checkpoint loading (E2E0, eval mode, device
placement), resampling arbitrary input sample rates to the 16kHz the model
was trained on, mel spectrogram extraction, 32-frame-aligned padding for the
U-Net's downsampling, and decoding the model's per-frame class distribution
to Hz (local-average or Viterbi). This is the one class most callers need.

Reads: model.py (E2E0/E2E), spec.py (MelSpectrogram), utils.py (F0 decoders),
constants.py (SAMPLE_RATE et al.).
"""

import torch
import torch.nn.functional as F
from torchaudio.transforms import Resample
from .constants import MEL_FMAX, MEL_FMIN, N_MELS, SAMPLE_RATE, WINDOW_LENGTH
from .model import E2E0
from .spec import MelSpectrogram
from .utils import to_local_average_f0, to_viterbi_f0


def resolve_device(device=None) -> torch.device:
    """Resolve legacy automatic selection and validate explicit devices."""
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        resolved = torch.device(device)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(
            "device must be None, 'auto', 'cpu', 'cuda', 'cuda:N', or 'mps'"
        ) from exc
    if resolved.type == "cpu" and resolved.index is None:
        return resolved
    if resolved.type == "mps" and resolved.index is None:
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("MPS was explicitly requested but is not available")
        return resolved
    if resolved.type != "cuda":
        raise ValueError("device must be None, 'auto', 'cpu', 'cuda', 'cuda:N', or 'mps'")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is not available")
    if resolved.index is not None and resolved.index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device index {resolved.index} is not available")
    return resolved


class RMVPE:
    """Robust Model for Vocal Pitch Estimation.

    Args:
        model_path: Path to the .pt checkpoint file.
        hop_length: Hop size in samples at 16kHz (default: 160 = 10ms).
        device: Device to run on ('cuda', 'cpu', or None/'auto' for auto-detect).
    """

    def __init__(self, model_path, hop_length=160, device=None):
        self.device = resolve_device(device)

        model = E2E0(4, 1, (2, 2))
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        model.eval()
        model.to(self.device)

        self.hop_length = hop_length
        self.seg_length = 32 * hop_length
        self.model = model
        self.mel_extractor = MelSpectrogram(
            N_MELS, SAMPLE_RATE, WINDOW_LENGTH, hop_length, None, MEL_FMIN, MEL_FMAX
        ).to(self.device)
        self.resample_kernel = {}

    def mel2hidden(self, mel):
        with torch.no_grad():
            n_frames = mel.shape[-1]
            mel = F.pad(mel, (0, 32 * ((n_frames - 1) // 32 + 1) - n_frames), mode="reflect")
            hidden = self.model(mel)
            return hidden[:, :n_frames]

    def decode(self, hidden, thred=0.03, use_viterbi=False):
        if use_viterbi:
            return to_viterbi_f0(hidden, thred=thred)
        return to_local_average_f0(hidden, thred=thred)

    def infer_from_audio(self, audio, sample_rate=16000, device=None, thred=0.03, use_viterbi=False):
        """Run pitch estimation on an audio array.

        Args:
            audio: 1-D numpy array of audio samples.
            sample_rate: Sample rate of the input audio.
            device: Ignored (uses device set at init). Kept for API compatibility.
            thred: Voicing confidence threshold (default: 0.03).
            use_viterbi: Use Viterbi decoding for smoother tracks.

        Returns:
            numpy array of F0 values in Hz (0 = unvoiced).
        """
        dev = self.device
        audio = torch.from_numpy(audio).float().unsqueeze(0).to(dev)

        if sample_rate != 16000:
            key_str = str(sample_rate)
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = Resample(sample_rate, 16000, lowpass_filter_width=128).to(dev)
            audio = self.resample_kernel[key_str](audio)

        B, T = audio.shape
        n_frames = T // self.hop_length + 1
        T1 = T + self.hop_length
        T_pad = self.seg_length * ((T1 - 1) // self.seg_length + 1) - T1
        audio = F.pad(audio, (0, T_pad))

        mel = self.mel_extractor(audio, center=True)
        with torch.no_grad():
            hidden = self.model(mel)

        return self.decode(hidden[:, :n_frames], thred=thred, use_viterbi=use_viterbi)
