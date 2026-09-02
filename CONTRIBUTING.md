# Contributing

## Development setup

CELNN keeps the existing Conda development environment for compatibility while
`pyproject.toml` remains the package manifest for installable dependencies and
optional capabilities.

Create the maintained development environment and install the package with the
Torch extra required by the package-wide Pyright analysis:

```bash
conda env create -f environment.yml
conda activate pycelnn
python -m pip install -e ".[torch]"
```

To update an existing environment:

```bash
conda env update -f environment.yml --prune
python -m pip install -e ".[torch]"
```

## Development workflow

1. Create a feature branch.
2. Make focused changes with matching tests.
3. Run `ruff check .`.
4. Run `pyright`.
5. Run `pytest`.
6. Open a pull request with a concise description of the change and its motivation.

## Full non-CUDA integration suite

The default workflow does not require every optional execution capability. To
reproduce the CI lane that exercises all supported non-CUDA integrations:

```bash
python -m pip install -e ".[torch,scipy,ga,image,viz]"
pytest
```

Real CuPy/CUDA behavior requires a CUDA-capable environment and is not implied
by the non-CUDA integration suite.

## Documentation module

Documentation tooling is isolated under `docs/` and is not part of the CELNN
runtime package or PyPI dependency metadata. Use Python 3.12 and install the
library plus the standalone docs project:

```bash
python -m pip install -e .
python -m pip install ./docs
python -m sphinx -W --keep-going -b html docs/source docs/_build/html
```

To validate Torch-backed API documentation with the real optional dependency:

```bash
python -m pip install -e ".[torch]"
python -m pip install ./docs
CELNN_DOCS_REAL_TORCH=1 \
  python -m sphinx -W --keep-going -b html \
  docs/source docs/_build/html-torch
```

The base documentation build may mock Torch for structural API rendering only.
Mocks are never evidence of Torch runtime behavior. Notebook execution errors
are fatal unless an example explicitly marks the exception as expected.

See [`docs/README.md`](docs/README.md) for the documentation-module contract.

## Style guidelines

- Keep the core package independent from optional image and plotting dependencies.
- Prefer readable numerical code over compact but opaque implementations.
- Add docstrings to public classes and functions.
- Raise `celnn` exceptions with explicit, actionable messages.

## Testing

The test suite is written with `pytest`. Optional integration tests skip cleanly
when their dependency is unavailable; CI exercises the base package, the
supported non-CUDA integrations, declared dependency floors, and built
artifacts separately.
