# Contributing

## Development setup

`pyproject.toml` is the source of truth for Python dependencies, optional
capabilities, and development tools. Install the package with the minimal
extras needed for the package-wide static checks:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev,torch]"
```

The `torch` extra is included because the maintained public implementation
contains optional `torch.nn.Module` subclasses that Pyright analyzes when
running the package-wide static gate.

## Development workflow

1. Create a feature branch.
2. Make focused changes with matching tests.
3. Run `ruff check .`.
4. Run `pyright`.
5. Run `pytest`.
6. Open a pull request with a concise description of the change and its motivation.

## Full non-CUDA integration suite

The default workflow intentionally does not install every optional capability.
To reproduce the CI lane that exercises all supported non-CUDA integrations:

```bash
python -m pip install -e ".[dev,torch,scipy,ga,image,viz]"
pytest
```

Real CuPy/CUDA behavior requires a CUDA-capable environment and is not implied
by the non-CUDA integration suite.

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
