"""Import smoke test -- the org constitution's minimum bar for every repo.

No weights, no network: just importability of the public surface and that
the version is single-sourced correctly.
"""

import rmvpe_infer


def test_package_imports():
    assert rmvpe_infer.RMVPE is not None
    assert rmvpe_infer.download_model is not None


def test_version_is_a_nonempty_string():
    assert isinstance(rmvpe_infer.__version__, str)
    assert rmvpe_infer.__version__


def test_version_matches_about_module():
    from rmvpe_infer.__about__ import __version__ as about_version

    assert rmvpe_infer.__version__ == about_version


def test_submodules_import():
    import rmvpe_infer.cli  # noqa: F401
    import rmvpe_infer.constants  # noqa: F401
    import rmvpe_infer.deepunet  # noqa: F401
    import rmvpe_infer.download  # noqa: F401
    import rmvpe_infer.inference  # noqa: F401
    import rmvpe_infer.model  # noqa: F401
    import rmvpe_infer.seq  # noqa: F401
    import rmvpe_infer.spec  # noqa: F401
    import rmvpe_infer.utils  # noqa: F401
