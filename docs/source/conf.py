# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))

project = "Posture Experiment"
copyright = "2023, Andy Tam Yiu Chau"
author = "Andy Tam Yiu Chau"
release = "0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx_markdown_tables",
]


templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}


# -- Options for ;Autodoc output ---------------------------------------------
autodoc_mock_imports = [
    "torch",
    "torchvision",
    "pytorch_lightning",
    "numpy",
    "pandas",
    "mmseg",
    "semseg",
    "Cython",
    "cpu_nms",
    "gpu_nms",
    "matplotlib",
    "pyccotools",
    "mmcv",
    "numpy.core.multiarray",
]
