import celnn


def test_top_level_public_api_names_are_well_formed_and_unique():
    assert len(celnn.__all__) == len(set(celnn.__all__))
    assert all(
        name.isidentifier() and not name.startswith("_")
        for name in celnn.__all__
    )


def test_runtime_version_comes_from_installed_metadata():
    assert celnn.__version__
