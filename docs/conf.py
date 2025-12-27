# Configuration file for the Sphinx documentation builder.

# -- Project information

project = "ADMET-XSpec"
copyright = "2025, Rybka, Masztalerz"
author = "Rybka, Masztalerz"

release = "1.0"
version = "1.0.0"

# -- General configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx_rtd_theme",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output
html_theme = "sphinx_rtd_theme"
