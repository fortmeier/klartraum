# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

# Sphinx configuration for the Klartraum Engine documentation.

import os
import re
import subprocess

DOCS_DIR = os.path.dirname(os.path.abspath(__file__))

# The API reference is generated from the public headers: Doxygen writes XML
# into _doxygen/, which Breathe renders. Running it here keeps a plain
# `sphinx-build docs <out>` sufficient, locally and in CI.
os.makedirs(os.path.join(DOCS_DIR, "_doxygen"), exist_ok=True)
subprocess.run(["doxygen", "Doxyfile"], cwd=DOCS_DIR, check=True)

project = "Klartraum Engine"
author = "Dirk Fortmeier"
copyright = "2025-2026, Dirk Fortmeier"

# The version is defined once, in project() in the top-level CMakeLists.txt.
with open(os.path.join(DOCS_DIR, "..", "CMakeLists.txt"), encoding="utf-8") as cmake_file:
    release = re.search(r"project\(klartraum VERSION ([0-9.]+)", cmake_file.read()).group(1)
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_parser",
    "breathe",
    "sphinx_needs",
]

breathe_projects = {"klartraum": os.path.join(DOCS_DIR, "_doxygen", "xml")}
breathe_default_project = "klartraum"
breathe_default_members = ("members",)

source_suffix = {".md": "markdown"}
exclude_patterns = ["_build", "_doxygen", ".venv"]

myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3

primary_domain = "cpp"
highlight_language = "cpp"

# Requirements, specifications and tests are sphinx-needs objects. IDs are
# upper case with a type prefix, e.g. REQ_GRAPH_001.
needs_types = [
    {"directive": "req", "title": "Requirement", "prefix": "REQ_", "color": "#BFD8D2", "style": "node"},
    {"directive": "spec", "title": "Specification", "prefix": "SPEC_", "color": "#FEDCD2", "style": "node"},
    {"directive": "test", "title": "Test", "prefix": "TEST_", "color": "#DCB239", "style": "node"},
]
needs_id_required = True
needs_id_regex = r"^(REQ|SPEC|TEST)_[A-Z0-9_]+$"
needs_fields = {
    "status": {
        "description": "draft, open, implemented or verified",
        "schema": {"type": "string", "enum": ["draft", "open", "implemented", "verified"]},
    },
    "gtest": {
        "description": "GoogleTest case that implements a test need, e.g. Suite.name",
        "schema": {"type": "string"},
        "nullable": True,
    },
}
needs_links = {
    "implements": {"incoming": "is implemented by", "outgoing": "implements"},
    "verifies": {"incoming": "is verified by", "outgoing": "verifies"},
}
needs_build_json = True

html_theme = "furo"
html_title = f"Klartraum Engine {release}"
html_static_path = []

# One Impressum and one privacy notice on klartraum.ai cover the landing page
# and both documentation sites; Furo renders footer_icons as plain links.
html_theme_options = {
    "footer_icons": [
        {
            "name": "Impressum",
            "url": "https://klartraum.ai/impressum.html",
            "html": "Impressum",
            "class": "",
        },
        {
            "name": "Datenschutz / Privacy",
            "url": "https://klartraum.ai/privacy.html",
            "html": "Datenschutz / Privacy",
            "class": "",
        },
    ],
}
