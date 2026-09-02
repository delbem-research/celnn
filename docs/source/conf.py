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
    "sphinx.ext.intersphinx",
]

source_suffix = {".md": "markdown"}
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

nitpicky = True

autodoc_member_order = "bysource"
_real_torch = os.environ.get("CELNN_DOCS_REAL_TORCH") == "1"
autodoc_mock_imports = [] if _real_torch else ["torch"]

myst_enable_extensions = ["amsmath", "colon_fence", "dollarmath"]

nb_execution_mode = "cache"
nb_execution_raise_on_error = True
nb_execution_timeout = 30
nb_execution_excludepatterns = ["labs/gpu/**", "labs/expensive/**"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
}
intersphinx_timeout = 10

html_theme = "furo"
html_title = f"CELNN {release} technical knowledge system"
