from __future__ import annotations

import os
from importlib.metadata import version as distribution_version

project = "celnn"
author = "CELNN contributors"
release = distribution_version("celnn")
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
]

master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

nitpicky = True
nitpick_ignore = [
    ("py:class", "dict[str"),
    ("py:class", "numpy._typing._dtype_like.DTypeLike"),
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
}

autodoc_member_order = "bysource"
autodoc_typehints = "signature"
_real_torch = os.environ.get("CELNN_DOCS_REAL_TORCH") == "1"
autodoc_mock_imports = [] if _real_torch else ["torch"]

myst_enable_extensions = ["amsmath", "colon_fence", "dollarmath"]

nb_execution_mode = "cache"
nb_execution_raise_on_error = True
nb_execution_timeout = 30

html_theme = "furo"
html_title = f"CELNN {release} technical knowledge system"
