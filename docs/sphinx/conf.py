import os
import sys
from datetime import datetime

# Add project src/ to sys.path so autodoc can import project packages
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

project = 'PHISE'
author = 'Vincent Foriel'
year = datetime.now().year

extensions = [
    'myst_parser',
    'sphinx_copybutton',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '.venv', 'index.rst', 'README.md']

html_theme = 'breeze'
html_static_path = ['_static']

autodoc_member_order = 'bysource'
# Éviter la duplication des sections Paramètres:
# - on laisse les types uniquement dans la signature,
# - et on confie à Napoleon la transformation des sections Parameters/Returns.
autodoc_typehints = 'signature'
autodoc_default_options = {
    'members': True,
    'show-inheritance': True,
}

# Pour une génération complète (CI), installez les dépendances listées dans
# requirements-docs.txt afin d'éviter d'avoir à moquer les imports.
# Ajoutez ici les libs externes non indispensables à la signature/type hints
# mais lourdes ou absentes de l'environnement de build.
autodoc_mock_imports = [
    'numba',
    'matplotlib',
    'scipy',
    'git',
    'astropy',
    'numpy',
]

# Napoleon settings: activer Google ET NumPy pour couvrir les deux styles
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

# myst-parser options (allow LaTeX math and some extensions)
myst_enable_extensions = [
    'amsmath', 'colon_fence', 'deflist', 'dollarmath', 'fieldlist',
    'html_admonition', 'html_image', 'linkify', 'replacements',
    'smartquotes', 'strikethrough', 'substitution', 'tasklist'
]

# Autoriser les "fenced code blocks" à agir comme des directives Sphinx
myst_fence_as_directive = [
    'autoclass', 'automodule', 'autofunction', 'autodata', 'autoexception', 'autosummary'
]

# HTML settings ---------------------------------------------------------------

html_logo = "../phise_logo.png"
html_title = "PHISE"
html_context = {
    "github_user": "vforiel",
    "github_repo": "Tunable-Kernel-Nulling",
    "github_version": "main",
    "doc_path": "docs",
    "version_switcher": "https://raw.githubusercontent.com/aksiome/breeze/refs/heads/main/docs/_static/switcher.json",
    
}

html_theme_options = {
    "emojis_header_nav": True,
}