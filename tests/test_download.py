"""Unit tests for download.py's checksum verification, env-var override, and
cache/extraction flow -- all offline, no real network call and no real
checkpoint required (the real download IS exercised indirectly by
test_pitch_physics.py / test_baseline_regression.py, which need the real
weights and are marked `weights`).
"""

import hashlib
import zipfile

import pytest

from rmvpe_infer import download as download_module
from rmvpe_infer.download import (
    ChecksumMismatchError,
    ENV_VAR,
    download_model,
    verify_checksum,
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class TestChecksumVerification:
    def test_matching_checksum_does_not_raise(self, tmp_path):
        fake = tmp_path / "fake.pt"
        fake.write_bytes(b"tiny fake checkpoint contents")
        expected = _sha256_bytes(fake.read_bytes())
        verify_checksum(fake, expected=expected)  # should not raise

    def test_mismatched_checksum_raises(self, tmp_path):
        fake = tmp_path / "fake.pt"
        fake.write_bytes(b"tiny fake checkpoint contents")
        with pytest.raises(ChecksumMismatchError):
            verify_checksum(fake, expected="0" * 64)

    def test_error_message_names_the_file(self, tmp_path):
        fake = tmp_path / "fake.pt"
        fake.write_bytes(b"x")
        with pytest.raises(ChecksumMismatchError, match=str(fake)):
            verify_checksum(fake, expected="0" * 64)


class TestEnvVarOverride:
    def test_env_var_short_circuits_download(self, tmp_path, monkeypatch):
        real_weights = tmp_path / "my_local_rmvpe.pt"
        real_weights.write_bytes(b"a local checkpoint the user already has")
        monkeypatch.setenv(ENV_VAR, str(real_weights))

        def _fail_if_called(*args, **kwargs):
            raise AssertionError("urlretrieve was called despite env var override")

        monkeypatch.setattr(download_module, "urlretrieve", _fail_if_called)

        result = download_model()
        assert result == real_weights

    def test_env_var_pointing_at_missing_file_raises(self, tmp_path, monkeypatch):
        missing = tmp_path / "does_not_exist.pt"
        monkeypatch.setenv(ENV_VAR, str(missing))
        with pytest.raises(FileNotFoundError):
            download_model()


class TestDownloadAndExtractFlow:
    """Exercises download_model's cache/extract logic with urlretrieve mocked
    out entirely -- no real network traffic, no real 350MB checkpoint."""

    @pytest.fixture(autouse=True)
    def _no_ambient_env_override(self, monkeypatch):
        """These tests exercise the cache_dir/download path directly, so they
        must not be short-circuited by a developer's own RMVPE_INFER_WEIGHTS
        (which CLAUDE.md/README recommend setting for local weight-marked
        tests) leaking in from the ambient environment."""
        monkeypatch.delenv(ENV_VAR, raising=False)

    def _fake_zip_with_pt(self, dest_zip_path, pt_name="model.pt", pt_contents=b"fake weights"):
        with zipfile.ZipFile(dest_zip_path, "w") as zf:
            zf.writestr(pt_name, pt_contents)

    def test_fresh_download_extracts_and_returns_pt_path(self, tmp_path, monkeypatch):
        cache_dir = tmp_path / "cache"

        def _fake_urlretrieve(url, filename):
            self._fake_zip_with_pt(filename)

        monkeypatch.setattr(download_module, "urlretrieve", _fake_urlretrieve)

        result = download_model(cache_dir=cache_dir, verify=False)

        assert result.name == "model.pt"
        assert result.read_bytes() == b"fake weights"
        assert not (cache_dir / "rmvpe.zip").exists()  # zip cleaned up

    def test_cache_hit_skips_download(self, tmp_path, monkeypatch):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True)
        (cache_dir / "model.pt").write_bytes(b"already cached")

        calls = []
        monkeypatch.setattr(
            download_module, "urlretrieve", lambda *a, **k: calls.append((a, k))
        )

        result = download_model(cache_dir=cache_dir, verify=False)

        assert calls == []  # urlretrieve never called
        assert result.read_bytes() == b"already cached"

    def test_force_redownloads_even_if_cached(self, tmp_path, monkeypatch):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True)
        (cache_dir / "model.pt").write_bytes(b"stale cached copy")

        def _fake_urlretrieve(url, filename):
            self._fake_zip_with_pt(filename, pt_contents=b"fresh redownloaded copy")

        monkeypatch.setattr(download_module, "urlretrieve", _fake_urlretrieve)

        result = download_model(cache_dir=cache_dir, force=True, verify=False)
        assert result.read_bytes() == b"fresh redownloaded copy"

    def test_verify_true_raises_on_bad_checksum(self, tmp_path, monkeypatch):
        cache_dir = tmp_path / "cache"

        def _fake_urlretrieve(url, filename):
            self._fake_zip_with_pt(filename)

        monkeypatch.setattr(download_module, "urlretrieve", _fake_urlretrieve)

        with pytest.raises(ChecksumMismatchError):
            download_model(cache_dir=cache_dir, verify=True)  # fake bytes != MODEL_SHA256
