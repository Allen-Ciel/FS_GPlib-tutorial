# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import datetime
import importlib
import inspect
import os
import sys

sys.path.insert(0, os.path.abspath('../../src'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'FS_GPlib-tutorial'
author = 'Allen Guo'
release = 'v0.1.0'
copyright = f'{datetime.datetime.now().year}, {author}'

# -- Internationalization ----------------------------------------------------

language = os.environ.get('SPHINX_LANGUAGE', 'en')
locale_dirs = ['locale/']
gettext_compact = False
gettext_uuid = True
gettext_location = True

LANGUAGES = {
    'en': 'English',
    'zh_CN': '中文',
}

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
smartquotes = False
extensions = [
    'myst_parser',
    "sphinxcontrib.mermaid",
    "sphinx_copybutton",
    "sphinxcontrib.bibtex",
    'sphinx.ext.autodoc',
    'sphinx.ext.linkcode',
]

autodoc_mock_imports = ['torch', 'torch_geometric', 'numpy', 'tqdm', 'torch_scatter']


def linkcode_resolve(domain, info):
    """Map documented Python objects to GitHub source URLs."""
    if domain != 'py' or not info['module']:
        return None
    try:
        mod = importlib.import_module(info['module'])
        obj = mod
        for part in info['fullname'].split('.'):
            obj = getattr(obj, part)
        fn = inspect.getfile(obj)
        src, lineno = inspect.getsourcelines(obj)
        end_lineno = lineno + len(src) - 1
    except Exception:
        return None
    rel = os.path.relpath(fn, start=os.path.abspath('../../'))
    return (
        f"https://github.com/{html_context['github_user']}"
        f"/{html_context['github_repo']}"
        f"/blob/{html_context['github_version']}/{rel}"
        f"#L{lineno}-L{end_lineno}"
    )

bibtex_bibfiles = ["refs.bib"]
bibtex_default_style = "unsrt"

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

myst_enable_extensions = [
    "tasklist",
    "deflist",
    "dollarmath",
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'analytics_anonymize_ip': False,
    'logo_only': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 5,
    'includehidden': True,
    'titles_only': False,
}

html_context = {
    "display_github": True,
    "github_user": "Allen-Ciel",
    "github_repo": "FS_GPlib-tutorial",
    "github_version": "docs",
    "conf_py_path": "/source/",
    "current_language": language,
    "languages": LANGUAGES,
}
html_logo = "./_static/logo_v2.png"
html_static_path = ['_static']
html_js_files = [
    'my_custom.js',
]
