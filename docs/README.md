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

The strict build uses Sphinx intersphinx inventories for Python, NumPy, and
PyTorch. Local, CI, GitHub Pages, and Read the Docs builders therefore need
outbound HTTPS access to those documentation sites.

## Canonical local build

Run from the repository root:

```bash
python -m pip install -e .
python -m pip install -r docs/requirements.txt
python -m sphinx -W --keep-going -b html docs/source docs/_build/html
python docs/check_reference.py docs/_build/html/objects.inv
```

`check_reference.py` verifies the generated Sphinx inventory rather than the
source markup. Every symbol intentionally exported through `celnn.__all__`,
`celnn.backends.__all__`, and `celnn.training.__all__` must exist exactly once
as a Python-domain object in the rendered reference. Canonical implementation
names that Sphinx inventories as aliases of a public object are accepted only
when they resolve to the same generated URI. A separately rendered private
CELNN target is rejected as implementation leakage.

The documentation source lives in `docs/source/`; generated output and notebook
cache stay under `docs/_build/` and are not committed.

## Real Torch validation and publication

The base documentation build mocks Torch only to prove that the reference can
still be generated without installing optional CELNN capabilities. Runtime
claims and published documentation use the real dependency.

For local real-Torch validation:

```bash
python -m pip install -e ".[torch]"
python -m pip install -r docs/requirements.txt
CELNN_DOCS_REAL_TORCH=1 \
  python -m sphinx -W --keep-going -b html \
  docs/source docs/_build/html-torch
python docs/check_reference.py docs/_build/html-torch/objects.inv
```

`CELNN_DOCS_REAL_TORCH=1` is the host-neutral publication switch. In pull-request
CI, the `Release` lane builds both documentation modes against the same exact
wheel: first without Torch as the optional-dependency isolation oracle, then
with CPU Torch for the complete public reference. The `Integrations` lane owns
runtime integrations only and installs no documentation tooling. GitHub Pages
uses the same real-Torch switch for publication. Read the Docs remains
compatible through its standard `READTHEDOCS=True` environment.

### GitHub Pages

`.github/workflows/docs-pages.yml` is the publication path for the latest docs.
It runs only after a push to `main`, installs the CPU-only Torch wheel, performs
the same strict Sphinx build and inventory check, and deploys the generated HTML
through GitHub Pages. It does not add a pull-request check or a package runtime
dependency.

The expected project-site URL is `https://delbem-research.github.io/celnn/` once
GitHub Pages is enabled with GitHub Actions as its publishing source.

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
