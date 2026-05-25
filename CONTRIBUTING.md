# Contributing

## Development setup

```bash
conda env create -f environment.yml
conda activate pycelnn
pip install -e . --no-deps
```

To update an existing environment:

```bash
conda env update -f environment.yml --prune
```

## Development workflow

1. Create a feature branch.
2. Make focused changes with tests.
3. Run `python -m compileall src`.
4. Run `pytest`.
5. Run `ruff check .`.
6. Run `mypy src/celnn/core/network.py src/celnn/core/simulation.py src/celnn/core/solvers.py`.
7. Open a pull request with a concise description of the change and its motivation.

## Style guidelines

- Keep the core package independent from optional image and plotting dependencies.
- Prefer readable numerical code over compact but opaque implementations.
- Add docstrings to public classes and functions.
- Raise `celnn` exceptions with explicit, actionable messages.

## Testing

The test suite is written with `pytest`. GPU runtime tests also skip
cleanly when CuPy/CUDA are unavailable.

```bash
conda activate pycelnn
pytest
```
