# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.

# import sphinx_gallery
from datetime import date

from numpydoc import docscrape
import sphinx_bootstrap_theme

import borsar


# -- Project information -----------------------------------------------------

project = 'borsar'
_today = date.today()
iso_date = _today.isoformat()
copyright = f'2018-{_today.year}, Mikołaj Magnuski. Last updated {iso_date}'
author = 'Mikołaj Magnuski'

# The short X.Y version
version = borsar.__version__
# The full version, including alpha/beta/rc tags
release = version

# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx_gallery.gen_gallery',
    'numpydoc',
]

# generate autosummary even if no references
autosummary_generate = True

# set autodoc
autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "show-inheritance": True,
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# The language for content autogenerated by Sphinx.
language = 'en'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = None


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'bootstrap'
html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    'navbar_sidebarrel': False,
    'navbar_links': [
        ("Examples", "auto_examples/index"),
        ("API", "api"),
        ("GitHub", "https://github.com/mmagnuski/borsar", True)
    ],
    'bootswatch_theme': "united"
}

html_templates_path = ['_templates']

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}


# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'borsardoc'


# -- Options for LaTeX output ------------------------------------------------

latex_elements = {}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'borsar.tex', 'borsar Documentation',
     'Mikołaj Magnuski', 'manual'),
]


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'borsar', 'borsar Documentation',
     [author], 1)
]


# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'borsar', 'borsar Documentation',
     author, 'borsar', 'One line description of project.',
     'Miscellaneous'),
]


# -- Options for Epub output -------------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project

# A list of files that should not be packed into the epub file.
epub_exclude_files = ['search.html']


# -- Extension configuration -------------------------------------------------

# -- Options for intersphinx extension ---------------------------------------

# configuration for intersphinx: refer to the Python standard library.
# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {'python': ('https://docs.python.org/3', None),
                       'mne': ('https://mne.tools/dev', None),
                       'numpy': ('https://numpy.org/devdocs', None),
                       'scipy': ('https://scipy.github.io/devdocs', None),
                       'mayavi': (
                           'http://docs.enthought.com/mayavi/mayavi',
                           None),
                       }

sphinx_gallery_conf = {
    'examples_dirs': '../examples',
    'gallery_dirs': 'auto_examples',
    'backreferences_dir': 'generated',
    'reference_url': {  # noqa: E501 --> https://sphinx-gallery.github.io/stable/configuration.html#add-intersphinx-links-to-your-examples
        'borsar': None
    },
}

# -- numpydoc ----------------------------------------------------------------
docscrape.ClassDoc.extra_public_methods = (
    '__getitem__', '__iter__', '__len__')
# https://stackoverflow.com/a/34604043/5201771
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False
numpydoc_xref_param_type = True
numpydoc_attributes_as_param_list = False
numpydoc_xref_ignore = {
    # words
    "of",
    "or",
    "shape",
    "optional",
}
