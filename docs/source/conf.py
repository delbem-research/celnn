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

# Keep the intersphinx capability available without making the strict core docs
# build depend on remote inventory availability. Add mappings only when a page
# introduces a semantic upstream reference that needs one.
intersphinx_mapping: dict[str, tuple[str, str | None]] = {}

html_theme = "furo"
html_title = f"CELNN {release} technical knowledge system"
