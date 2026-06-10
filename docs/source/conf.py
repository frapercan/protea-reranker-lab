# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from pathlib import Path

# -- Path setup ---------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

# -- Project information ------------------------------------------------------
project = "protea-reranker-lab"
author = "Francisco Miguel Perez Canales"
copyright = "2026, Francisco Miguel Perez Canales"
release = "0.3.0"

# -- General configuration ----------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "myst_parser",
]

# MyST lets the published ADR / provenance Markdown files render inside
# the Sphinx tree without conversion to reStructuredText.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Mock heavy runtime dependencies so autodoc never fails on import ---------
autodoc_mock_imports = [
    "lightgbm",
    "numpy",
    "pandas",
    "pyarrow",
    "pyarrow.compute",
    "pyarrow.dataset",
    "pyarrow.parquet",
    "sklearn",
    "sklearn.model_selection",
    "sklearn.preprocessing",
    "wandb",
    "yaml",
    "protea_contracts",
    "protea_contracts.contexts",
    "matplotlib",
    "matplotlib.pyplot",
]

# autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "special-members": "__init__",
}
autodoc_typehints = "description"
autodoc_typehints_format = "short"
add_module_names = False

# Suppress known duplicate-target warnings caused by DatasetSpec being
# re-exported from both contracts and schemas.
suppress_warnings = ["ref.python"]

# napoleon (Google/NumPy docstring styles)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_preprocess_types = True
# Render dataclass ``Attributes:`` sections as info-field lists rather than
# standalone attribute directives, so they do not duplicate the attribute
# descriptions autodoc already emits for the same fields.
napoleon_use_ivar = True

# intersphinx mapping for standard library
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
}

# -- Options for HTML output --------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = []
