#!/usr/bin/env bash

# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

# Assembles the GitHub Pages site in _site/: the landing page from site/ at the
# root and the Sphinx documentation under docs/. Run from the repository root.
# SPHINX_BUILD selects the sphinx-build executable (default: docs/.venv, then PATH).
set -euo pipefail

cd "$(dirname "$0")/../.."

SPHINX_BUILD="${SPHINX_BUILD:-}"
if [ -z "$SPHINX_BUILD" ]; then
    if [ -x docs/.venv/bin/sphinx-build ]; then
        SPHINX_BUILD=docs/.venv/bin/sphinx-build
    else
        SPHINX_BUILD=sphinx-build
    fi
fi

rm -rf _site
mkdir -p _site
cp -R site/. _site/
"$SPHINX_BUILD" -W --keep-going -q docs _site/docs
# Pages would otherwise run Jekyll over the output and drop _static/ and _images/.
touch _site/.nojekyll

echo "site assembled in _site/"
