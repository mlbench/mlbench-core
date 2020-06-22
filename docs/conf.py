#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# Imports
#

import os
import sys
from os.path import abspath, dirname, join
from unittest.mock import MagicMock

sys.path.insert(0, abspath(join(dirname(__file__), ".")))
sys.path.insert(0, abspath(join(dirname(__file__), "..")))


class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()


MOCK_MODULES = [
    "torch",
    "torch.distributed",
    "torch.nn",
    "torch.nn.functional",
    "torch.nn.modules",
    "torch.nn.modules.loss",
    "torch.utils",
    "torch.utils.model_zoo",
    "torch.nn.init",
    "torch.utils.data",
    "torch.optim",
    "torch.optim.lr_scheduler",
    "torchvision",
    "torchvision.datasets",
    "torchvision.transforms",
]
sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)

# -- RTD configuration ------------------------------------------------

# on_rtd is whether we are on readthedocs.org, this line of code grabbed from docs.readthedocs.org
on_rtd = os.environ.get("READTHEDOCS", None) == "True"

# This is used for linking and such so we link to the thing we're building
rtd_version = os.environ.get("READTHEDOCS_VERSION", "latest")
if rtd_version not in ["stable", "latest"]:
    rtd_version = "stable"

# -- General configuration ------------------------------------------------


# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinxcontrib.napoleon",
    "sphinxcontrib.bibtex",
    "autoapi.extension",
]

autoapi_dirs = ["../mlbench_core"]
autoapi_generate_api_docs = False
autoapi_ignore = ["*migrations*", "*/preprocess/*.py"]
# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]
html_static_path = ["_static"]

# The suffix of source filenames.
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

autodoc_mock_imports = ["torch"]

# General information about the project.
project = "MLBench Core"
copyright = "2018 MLBench development team"

# napoleon
napoleon_google_docstring = True
napoleon_numpy_docstring = False


autoapiclass_content = "both"

intersphinx_mapping = {}

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for HTML output ----------------------------------------------

if not on_rtd:  # only import and set the theme if we're building docs locally
    import sphinx_rtd_theme

    html_theme = "sphinx_rtd_theme"
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
    print(html_theme_path)
else:
    html_theme = "default"

html_theme_options = {"navigation_depth": 5}

# Output file base name for HTML help builder.
htmlhelp_basename = "MLBench_Core"


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (
        "index",
        "MLBench_DaMLBench_Coreshboard.tex",
        "MLBench Core Documentation",
        "MLBench development team",
        "manual",
    ),
]

# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        "index",
        "MLBench_Core",
        "MLBench Core Documentation",
        "MLBench Core development team",
        "MLBench_Core",
        "One line description of project.",
        "Miscellaneous",
    ),
]

# -- Options for Epub output ----------------------------------------------

# Bibliographic Dublin Core info.
epub_title = "MLBench Core"
epub_author = "MLBench development team"
epub_publisher = "MLBench development team"
epub_copyright = "2018, MLBench development team"

# A list of files that should not be packed into the epub file.
epub_exclude_files = ["search.html"]

# -- Custom Document processing ----------------------------------------------

import gensidebar  # isort:skip

gensidebar.generate_sidebar(globals(), "mlbench_core")

import sphinx.addnodes  # isort:skip
import docutils.nodes  # isort:skip


def process_child(node):
    """This function changes class references to not have the
       intermediate module name by hacking at the doctree"""

    # Edit descriptions to be nicer
    if isinstance(node, sphinx.addnodes.desc_addname):
        if len(node.children) == 1:
            child = node.children[0]
            text = child.astext()

    # Edit literals to be nicer
    elif isinstance(node, docutils.nodes.literal):
        child = node.children[0]
        text = child.astext()

    for child in node.children:
        process_child(child)


def doctree_read(app, doctree):
    for child in doctree.children:
        process_child(child)


def setup(app):
    print("Yay!")
    app.add_css_file("css/custom.css")
    app.connect("doctree-read", doctree_read)
