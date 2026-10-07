#!/bin/sh

# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

# Adds the SPDX copyright and license header to every tracked first-party
# source file. Files that cannot carry a comment, and documentation and
# configuration, are covered by REUSE.toml instead. Files that already have a
# header are left alone, so the script can be run again at any time, e.g. on a
# branch before rebasing it.
#
# Usage, from anywhere in the repository: scripts/dev/reuse_annotate.sh

set -eu
cd "$(git rev-parse --show-toplevel)"

annotate() {
    style=$1
    shift
    git ls-files -z -- "$@" ':(exclude)3rdparty/**' |
        xargs -0 uvx --from 'reuse[charset-normalizer]' reuse annotate \
            --copyright "Dirk Fortmeier" --exclude-year --license MIT \
            --style "$style" --skip-existing
}

annotate cppsingle '*.cpp' '*.hpp' '*.inc' '*.comp' '*.vert' '*.frag' '*.mesh' '*.glsl' '*.mjs'
annotate python '*.py' '*.sh' 'CMakeLists.txt' '*/CMakeLists.txt'
