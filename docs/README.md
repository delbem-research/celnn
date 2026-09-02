# CELNN documentation module

This directory is a standalone documentation/tooling project. It depends on an
installed `celnn` package for API introspection and executable examples, but the
library never depends on this directory.

The root Python package intentionally does not expose a `docs` extra. Deleting
this directory must not change the runtime or PyPI metadata of `celnn`.

## Local build

Use Python 3.12 from the repository root:

```bash
python -m pip install -e .
python -m pip install ./docs
python docs/check_reference.py
python -m sphinx -W --keep-going -b html docs/source docs/_build/html
```

The reference check enforces that every top-level `celnn.__all__` export is
represented exactly once in the generated-reference source. The documentation
source lives in `docs/source/`; generated output and notebook cache stay under
`docs/_build/` and are not committed.

## Real Torch validation

The base documentation build mocks Torch only to render the structure of
Torch-backed public APIs. Runtime claims must use the real dependency:

```bash
python -m pip install -e ".[torch]"
python -m pip install ./docs
CELNN_DOCS_REAL_TORCH=1 \
  python -m sphinx -W --keep-going -b html \
  docs/source docs/_build/html-torch
```

Mocked imports are reference-rendering aids, never evidence that optional
behavior works.
