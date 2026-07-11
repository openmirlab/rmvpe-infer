"""Single source of truth for the package version.

Read by both `pyproject.toml` (via `[tool.hatch.version] path = ...`, so the
built wheel/sdist metadata always matches) and `rmvpe_infer/__init__.py` (so
`rmvpe_infer.__version__` matches too). Previously the version was a single
hand-maintained literal in both `pyproject.toml` and `__init__.py` that could
drift out of sync — this file is the only place it is written.
"""

__version__ = "0.1.0"
