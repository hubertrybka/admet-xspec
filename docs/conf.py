# Configuration file for the Sphinx documentation builder.

# Make conf.py resolve our module
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#module-sphinx.ext.autodoc
import sys
from pathlib import Path

sys.path.insert(0, str(Path("..").resolve()))


# -- Project information

project = "ADMET-XSpec"
copyright = "2025, Rybka, Masztalerz"
author = "Rybka, Masztalerz"

release = "1.0"
version = "1.0.0"

# -- General configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx_rtd_theme",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,
}
