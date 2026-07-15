"""Download and verify the pretrained RMVPE model checkpoint.

Fetches the official yxlllc/RMVPE release archive, extracts the .pt
checkpoint, and verifies its sha256 against the recorded provenance hash
(see README's "Pretrained Model" section for the org-hosting follow-up
note — this URL is a third-party GitHub release, not yet an openmirlab
mirror). `RMVPE_INFER_WEIGHTS` lets callers (tests, air-gapped CI, users
with their own copy) point at an already-present checkpoint and skip the
network and cache logic entirely.

Reads: nothing else in this package (URL/sha256 are local constants).
"""

import hashlib
import os
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

MODEL_URL = "https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "rmvpe"
ENV_VAR = "RMVPE_INFER_WEIGHTS"

# sha256 of the extracted rmvpe.pt checkpoint from the 230917 release,
# recorded 2026-07-12 from a checkpoint already cached in this environment.
# Verified on every download AND every cache hit (see download_model) so a
# truncated/corrupted transfer or a silently-swapped mirror file fails loudly
# instead of loading a bad model.
MODEL_SHA256 = "19dc1809cf4cdb0a18db93441816bc327e14e5644b72eeaae5220560c6736fe2"


class ChecksumMismatchError(RuntimeError):
    """Raised when a checkpoint's sha256 doesn't match the recorded value."""


def _sha256(path, chunk_size: int = 1 << 20) -> str:
    """Stream a file's sha256 hex digest so large checkpoints don't need to fit in memory at once."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksum(path, expected: str = MODEL_SHA256) -> None:
    """Raise ChecksumMismatchError if `path`'s sha256 doesn't match `expected`."""
    actual = _sha256(path)
    if actual != expected:
        raise ChecksumMismatchError(
            f"{path}: sha256 {actual} does not match expected {expected} "
            "(download may be corrupted, truncated, or tampered — delete the "
            "cache and retry)"
        )


def download_model(cache_dir=None, force: bool = False, verify: bool = True,
                   url: str = MODEL_URL, expected_sha256: str = MODEL_SHA256) -> Path:
    """Return the path to the pretrained RMVPE .pt checkpoint, downloading it if needed.

    Honors `RMVPE_INFER_WEIGHTS` (path to an existing checkpoint) before
    touching the network or cache directory at all — set it to point at a
    local copy for offline use or tests. Set `verify=False` to skip the
    sha256 check (e.g. if you intentionally use a checkpoint that predates
    the recorded hash).
    """
    env_path = os.environ.get(ENV_VAR)
    if env_path:
        path = Path(env_path)
        if not path.exists():
            raise FileNotFoundError(f"${ENV_VAR}={env_path} does not exist")
        return path

    cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Check both possible names
    model_path = cache_dir / "model.pt"
    alt_path = cache_dir / "rmvpe.pt"
    if not model_path.exists() and alt_path.exists():
        model_path = alt_path

    if model_path.exists() and not force:
        if verify:
            verify_checksum(model_path, expected=expected_sha256)
        return model_path

    zip_path = cache_dir / "rmvpe.zip"
    print(f"Downloading RMVPE model to {cache_dir}...")
    urlretrieve(url, str(zip_path))

    print("Extracting...")
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(str(cache_dir))
    zip_path.unlink(missing_ok=True)

    # The zip might extract with a different name — find any .pt file
    if not model_path.exists():
        for f in cache_dir.glob("*.pt"):
            model_path = f
            break

    if verify:
        verify_checksum(model_path, expected=expected_sha256)

    print(f"Model ready: {model_path}")
    return model_path


def main():
    """CLI entry point for downloading the model."""
    import argparse
    parser = argparse.ArgumentParser(description="Download RMVPE pretrained model")
    parser.add_argument("--cache-dir", type=str, default=None, help="Cache directory")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    parser.add_argument("--no-verify", action="store_true", help="Skip sha256 verification")
    args = parser.parse_args()
    path = download_model(cache_dir=args.cache_dir, force=args.force, verify=not args.no_verify)
    print(f"Model path: {path}")
