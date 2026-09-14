"""Unit tests for config.py's checkpoint metadata loading -- offline, no
network or real checkpoint required.

Guards against the checkpoint catalog's `license` field regressing back to
an unverified placeholder (it previously said "Unknown checkpoint license;
verify upstream terms before redistribution" until verified 2026-09-14 --
see CLAUDE.md's "Checkpoint license" section).
"""

from rmvpe_infer.config import checkpoint_config, checkpoint_entry

_PLACEHOLDER_VALUES = {"", "unknown", "tbd", "todo"}

_OLD_UNVERIFIED_LICENSE = (
    "Unknown checkpoint license; verify upstream terms before redistribution"
)

_TOML_TEMPLATE = """\
[checkpoints.rmvpe]
url = "https://example.invalid/rmvpe.zip"
sha256 = "0000000000000000000000000000000000000000000000000000000000000000"
filename = "model.pt"
license = {license!r}
provenance = "test fixture"
source_revision = "test"
updated_at = "2026-01-01"
"""


def _is_placeholder_license(value: str) -> bool:
    """Shared predicate: is this a vague/unverified license value rather than
    a real, checked one (an SPDX id, or the explicit `NOASSERTION` marker for
    a confirmed no-license upstream)?
    """
    normalized = value.strip().lower()
    return normalized in _PLACEHOLDER_VALUES or "unknown" in normalized


class TestCheckpointLicenseField:
    def test_license_field_present_and_non_empty(self):
        entry = checkpoint_entry()
        assert "license" in entry
        assert entry["license"].strip() != ""

    def test_license_field_is_not_a_placeholder(self):
        assert not _is_placeholder_license(checkpoint_entry()["license"])

    def test_license_field_is_the_verified_noassertion_value(self):
        """This checkpoint (yxlllc/RMVPE release 230917) has no license grant
        from any upstream party -- verified against Dream-High/RMVPE (the
        Apache-2.0 original code repo) and yxlllc/RMVPE (the fork, carrying
        no LICENSE file, that actually trained and released the checkpoint).
        NOASSERTION is the honest, verified value, matching the sibling
        convention (scnet-infer's own checkpoints.toml uses the same marker
        for the same situation) -- never silently reverted to a fabricated
        license.
        """
        assert checkpoint_entry()["license"] == "NOASSERTION"

    def test_every_checkpoint_entry_has_a_verified_license_field(self):
        for model_key, entry in checkpoint_config().items():
            assert not _is_placeholder_license(entry.get("license", "")), model_key


class TestPlaceholderRegressionGuard:
    def test_old_unverified_placeholder_value_is_rejected(self, tmp_path):
        """Positive proof the check actually rejects the old value, not just
        a tautology about the string literal: load a fixture config carrying
        the exact placeholder this repo used to ship, through the real
        `checkpoint_entry` loader (`config_path` override), and confirm the
        placeholder predicate flags it.
        """
        config_path = tmp_path / "checkpoints.toml"
        config_path.write_text(
            _TOML_TEMPLATE.format(license=_OLD_UNVERIFIED_LICENSE)
        )
        entry = checkpoint_entry(config_path=config_path)
        assert _is_placeholder_license(entry["license"])

    def test_a_real_verified_value_is_accepted(self, tmp_path):
        """Sanity check the predicate isn't just rejecting everything."""
        config_path = tmp_path / "checkpoints.toml"
        config_path.write_text(_TOML_TEMPLATE.format(license="NOASSERTION"))
        entry = checkpoint_entry(config_path=config_path)
        assert not _is_placeholder_license(entry["license"])
