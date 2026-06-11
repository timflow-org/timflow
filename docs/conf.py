# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
# import os
# import sys

project = "timflow"
copyright = "2026, Mark Bakker, Davíd Brakenhoff"
author = "Mark Bakker, Davíd Brakenhoff"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "IPython.sphinxext.ipython_console_highlighting",  # lowercase didn't work
    "numpydoc",
    "myst_nb",
    "sphinx_design",
    "sphinx.ext.autosectionlabel",
    "sphinxcontrib.bibtex",
]

# templates_path = ["_templates"]
exclude_patterns = ["_build", "_templates", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_logo = "_static/timflow_logo.jpeg"
html_short_title = "timflow"
html_favicon = "_static/timflow_icon.png"
html_css_files = ["css/custom.css"]
html_show_sphinx = True
html_show_copyright = True
htmlhelp_basename = "timflow"  # Output file base name for HTML help builder.
html_use_smartypants = True
html_show_sourcelink = True

html_theme_options = {
    "github_url": "https://github.com/timflow-org/timflow",
    "use_edit_page_button": True,
    "header_links_before_dropdown": 6,
    "show_nav_level": 2,
}

html_context = {
    "github_user": "timflow-org",
    "github_repo": "timflow",
    "github_version": "main",
    "doc_path": "docs",
}

# -- Napoleon settings ----------------------------------------------------------------

napoleon_include_init_with_doc = False
napoleon_use_param = True

# -- Autosectionlabel settings --------------------------------------------------------

autosectionlabel_prefix_document = True

# -- AutoAPI settings -----------------------------------------------------------------
autoapi_dirs = ["../timflow/steady", "../timflow/transient", "../timflow/bessel"]
autoapi_root = "api"
autoapi_options = [
    "show-module-summary",
    "inherited-members",
    "show-inheritance",
]
autoapi_own_page_level = "class"
autoapi_template_dir = "_templates/autoapi"
suppress_warnings = ["autoapi"]

# Keep API signatures and section navigation compact by omitting module prefixes
# (e.g., show `River` instead of `timflow.steady.linesink.River`).
add_module_names = False

# Keep local object TOC entries compact in the right sidebar.
toc_object_entries_show_parents = "hide"

# -- Numpydoc settings ----------------------------------------------------------------

numpydoc_class_members_toctree = True
numpydoc_show_class_members = False

# -- Set intersphinx Directories ------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/devdocs", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}

# -- myst_nb options ------------------------------------------------------------------

nb_execution_mode = "cache"

nb_execution_allow_errors = True  # Allow errors in notebooks, to see the error online
nb_execution_show_tb = True
nb_execution_timeout = 100
nb_execution_excludepatterns = [
    "besselnumba_timing.ipynb",
    "vertical_anisotropy.ipynb",
    "transient/04pumpingtests/*.ipynb",
]

myst_enable_extensions = ["dollarmath", "amsmath", "html_image"]
myst_dmath_double_inline = True

nb_merge_streams = True


# -- bibtex options ------------------------------------------------------------------

# Add some settings for bibtex
bibtex_bibfiles = ["about/publications.bib"]
bibtex_reference_style = "author_year"
