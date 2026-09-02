# CELNN documentation module

This directory is a standalone documentation/tooling project. It depends on an
installed `celnn` package for API introspection and executable examples, but the
library never depends on this directory.

The root Python package intentionally does not expose a `docs` extra. Deleting
this directory must not change the runtime or PyPI metadata of `celnn`.

## Tooling

Use Python 3.12. Documentation-only dependencies are owned by
`docs/requirements.txt`; they are not package metadata and are not installed
with CELNN from PyPI.

The current tested ranges are:

- Sphinx `>=9.1,<10`
- MyST-NB `>=1.4,<2`
- Furo `>=2025.12.19,<2027`

## Canonical local build

Run from the repository root:

```bash
python -m pip install -e .
python -m pip install -r docs/requirements.txt
python docs/check_reference.py
python -m sphinx -W --keep-going -b html docs/source docs/_build/html
```

`check_reference.py` verifies every symbol intentionally exported through
`celnn.__all__`, `celnn.backends.__all__`, and `celnn.training.__all__` exactly
once across the reference sources. The checker scans the reference tree rather
than depending on a fixed page layout.

The documentation source lives in `docs/source/`; generated output and notebook
cache stay under `docs/_build/` and are not committed.

## Real Torch validation

The base documentation build mocks Torch only to render the structure of
Torch-backed public APIs. Runtime claims must use the real dependency:

```bash
python -m pip install -e ".[torch]"
python -m pip install -r docs/requirements.txt
CELNN_DOCS_REAL_TORCH=1 \
  python -m sphinx -W --keep-going -b html \
  docs/source docs/_build/html-torch
```

Mocked imports are reference-rendering aids, never evidence that optional
behavior works.

## Executable examples

Unexpected notebook execution errors are fatal. An example that intentionally
teaches an exception must mark that cell with MyST-NB's `raises-exception` tag
rather than weakening error handling globally.

Hardware-dependent benchmarks or expensive studies do not belong in the normal
documentation build. When such material is added, its execution policy must be
introduced explicitly with the material instead of pre-configuring unused
exclude paths.
