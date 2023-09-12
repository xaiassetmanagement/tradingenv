#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# tradingenv documentation build configuration file, created by
# sphinx-quickstart on Tue Sep 17 10:35:56 2019.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have basis_naive_tm1 default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# TODO: use autodoc_mock_imports = ['numpy'] instead of mocking modules.

import os
import sys
from unittest import mock
pkg_dir = os.path.abspath('../../')
sys.path.insert(0, pkg_dir)


# These lines added to enable tradingenv to work without installing
# dependencies.
def mock_gym():
    """A workaround to sys.modules['gym'] = mock.Mock() which cannot be
    used otherwise TradingEnv (which inherits gym.Env) will become a Mock
    instance and docstring will not be rendered."""
    class MockEnv: pass
    class MockSpace: pass
    class MockDiscrete: pass
    class MockDict: pass
    class MockBox: pass
    from collections import namedtuple
    MockSpaces = namedtuple('spaces', ['Discrete', 'Dict', 'Box'])
    MockGym = namedtuple('gym', ['Env', 'Space', 'spaces'])
    sys.modules['gym'] = MockGym(
        Env=MockEnv,
        Space=MockSpace,
        spaces=MockSpaces(MockDiscrete, MockDict, MockBox),
    )


MOCK_MODULES = [
    'tradingenv.spaces',
    'tradingenv.registry',
    'tradingenv.dashboard',
    'numpy',
    'pandas',
    'pandas.core',
    'pandas.core.generic',
    'pandas.tseries.offsets',
    'gym',
    'gym.spaces',
    'gym.spaces.Space',
    'plotly',
    'plotly.figure_factory',
    'plotly.graph_objs',
    'statsmodels',
    'statsmodels.api',
    'statsmodels.regression',
    'statsmodels.regression.linear_model',
    'scipy',
    'scipy.cluster',
    'scipy.cluster.hierarchy',
    'scipy.optimize',
    'scipy.special',
    'sklearn.exceptions',
    'tensorflow',
    'tqdm',
]
mock_gym()
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = mock.Mock()


from tradingenv import __version__, __package__

# -- General configuration ------------------------------------------------

# If your documentation needs basis_naive_tm1 minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module _names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    # One only of the following two to parse numpy docstrings.
    'sphinx.ext.napoleon',  # preferred wrt to 'numpydoc' because built in

              ]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['.templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as basis_naive_tm1 list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = __package__
copyright = '2019, Federico Fontana'
author = 'Federico Fontana'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = __version__
# The full version, including alpha/beta/rc tags.
release = __version__

# The language for content autogenerated by Sphinx. Refer to documentation
# for basis_naive_tm1 list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'env'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = []

# The id of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# basis_naive_tm1 list of builtin themes.
#
html_theme = 'alabaster'

# Theme options are theme-specific and customize the look and feel of basis_naive_tm1 theme
# further.  For basis_naive_tm1 list of options available for each theme, see the
# documentation.
#
html_theme_options = {'show_related': False}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so basis_naive_tm1 file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['.static']

# Custom sidebar templates, must be basis_naive_tm1 dictionary that maps document _names
# to template _names.
#
# This is required for the alabaster theme
# refs: http://alabaster.readthedocs.io/en/latest/installation.html#sidebars
html_sidebars = {
    '**': [
        'relations.html',  # needs 'show_related': True theme option to display
        'searchbox.html',
        'localtoc.html',
    ]
}


# -- Options for HTMLHelp output ------------------------------------------

# Output file base id for HTML help builder.
htmlhelp_basename = 'tradingenvdoc'

# https://stackoverflow.com/questions/5599254/how-to-use-sphinxs-autodoc-to-document-a-classs-init-self-method
# To make class.__init__ docstring appear in class docstring.
autoclass_content = 'both'
autosummary_generate = True

# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target id, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'tradingenv.tex', 'tradingenv Documentation',
     'Federico Fontana', 'manual'),
]


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, id, description, authors, manual section).
man_pages = [
    (master_doc, 'tradingenv', 'tradingenv Documentation',
     [author], 1)
]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target id, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'tradingenv', 'tradingenv Documentation',
     author, 'tradingenv', 'One line description of project.',
     'Miscellaneous'),
]
#html_theme_options = {"fixed_sidebar": True}

# Example configuration for intersphinx: refer to the Python standard library.
# intersphinx_mapping = {'https://docs.python.org/': None}

#class_members_toctree = False
#numpydoc_show_class_members = False
