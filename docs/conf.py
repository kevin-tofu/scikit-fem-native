from __future__ import annotations

from importlib.metadata import version as package_version


project = "skfem-native"
author = "skfem-native contributors"
release = package_version("skfem-native")
version = release

extensions = [
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
exclude_patterns = ["_build"]
html_theme = "furo"
html_title = f"skfem-native {release}"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#006b5f",
        "color-brand-content": "#007c6e",
        "font-stack": "Avenir Next, Segoe UI, sans-serif",
        "font-stack--monospace": "JetBrains Mono, Consolas, monospace",
    },
    "dark_css_variables": {
        "color-brand-primary": "#62d8c6",
        "color-brand-content": "#79e3d2",
    },
}
