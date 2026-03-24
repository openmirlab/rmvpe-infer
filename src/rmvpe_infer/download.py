"""Download pretrained RMVPE model checkpoint."""

import zipfile
from pathlib import Path
from urllib.request import urlretrieve

MODEL_URL = "https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "rmvpe"


def download_model(cache_dir: str | Path | None = None, force: bool = False) -> Path:
    """Download the pretrained RMVPE model.

    Returns the path to the .pt checkpoint file.
    """
    cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Check both possible names
    model_path = cache_dir / "model.pt"
    alt_path = cache_dir / "rmvpe.pt"
    if not model_path.exists() and alt_path.exists():
        model_path = alt_path

    if model_path.exists() and not force:
        return model_path

    zip_path = cache_dir / "rmvpe.zip"
    print(f"Downloading RMVPE model to {cache_dir}...")
    urlretrieve(MODEL_URL, str(zip_path))

    print("Extracting...")
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(str(cache_dir))
    zip_path.unlink(missing_ok=True)

    # The zip might extract with a different name — find any .pt file
    if not model_path.exists():
        for f in cache_dir.glob("*.pt"):
            model_path = f
            break

    print(f"Model ready: {model_path}")
    return model_path


def main():
    """CLI entry point for downloading the model."""
    import argparse
    parser = argparse.ArgumentParser(description="Download RMVPE pretrained model")
    parser.add_argument("--cache-dir", type=str, default=None, help="Cache directory")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    args = parser.parse_args()
    path = download_model(cache_dir=args.cache_dir, force=args.force)
    print(f"Model path: {path}")
