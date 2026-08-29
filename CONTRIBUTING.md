# Contributing

## Development setup

```bash
conda env create -f environment.yml
conda activate celnn
pip install -e . --no-deps
```

To update an existing environment:

```bash
conda env update -f environment.yml --prune
```

## Development workflow

1. Create a feature branch.
2. Make focused changes with tests.
3. Run `ruff check .`.
4. Run `pyright`.
5. Run `pytest`.
6. Open a pull request with a concise description of the change and its motivation.

## Style guidelines

- Keep the core package independent from optional image and plotting dependencies.
- Prefer readable numerical code over compact but opaque implementations.
- Add docstrings to public classes and functions.
- Raise `celnn` exceptions with explicit, actionable messages.

## Testing

The test suite is written with `pytest`. Optional integration tests skip cleanly when
their dependency is unavailable; the full CI lane installs the supported non-CUDA
extras explicitly.

```bash
conda activate celnn
pytest
```
