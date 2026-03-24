"""CLI for RMVPE inference."""

import argparse
import csv
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="RMVPE vocal pitch estimation")
    parser.add_argument("-i", "--input", required=True, help="Input audio file")
    parser.add_argument("-o", "--output", default="f0.csv", help="Output CSV file")
    parser.add_argument("-m", "--model", default=None, help="Model checkpoint path (auto-downloads if omitted)")
    parser.add_argument("--hop-length", type=int, default=160, help="Hop length in samples (default: 160 = 10ms)")
    parser.add_argument("--threshold", type=float, default=0.03, help="Voicing threshold")
    parser.add_argument("--viterbi", action="store_true", help="Use Viterbi decoding")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu, auto if omitted)")
    args = parser.parse_args()

    import librosa
    from .inference import RMVPE
    from .download import download_model

    model_path = args.model or str(download_model())
    rmvpe = RMVPE(model_path, hop_length=args.hop_length)

    audio, sr = librosa.load(args.input, sr=None, mono=True)
    f0 = rmvpe.infer_from_audio(audio, sample_rate=sr, device=args.device,
                                 thred=args.threshold, use_viterbi=args.viterbi)

    hop_sec = args.hop_length / 16000
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_s", "f0_hz"])
        for i, freq in enumerate(f0):
            writer.writerow([round(i * hop_sec, 4), round(float(freq), 2)])

    voiced = np.sum(f0 > 0)
    print(f"Wrote {len(f0)} frames ({voiced} voiced) to {args.output}")
