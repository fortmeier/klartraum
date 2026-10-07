# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

"""Checks that every header has an include guard named after its path.

    include/klartraum/computegraph/copybuffer.hpp -> KLARTRAUM_COMPUTEGRAPH_COPYBUFFER_HPP
    tests/test_scene.hpp                          -> KLARTRAUM_TESTS_TEST_SCENE_HPP

The guard is the first preprocessor directive of the file:

    #ifndef KLARTRAUM_..._HPP
    #define KLARTRAUM_..._HPP
    ...
    #endif // KLARTRAUM_..._HPP

Usage:
    python3 scripts/dev/include_guards.py [--fix] [FILE...]

Without files, all tracked .hpp files are checked. With --fix, missing or
misnamed guards and `#pragma once` are rewritten. Exit code 1 if a header is
(or was, before fixing) not as described.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

DIRECTIVE = re.compile(r"^\s*#\s*(\w+)\s*(\S*)")


def guard_name(path: str) -> str:
    parts = PurePosixPath(path).parts
    if parts[:2] == ("include", "klartraum"):
        parts = parts[2:]
    return "KLARTRAUM_" + re.sub(r"[^A-Za-z0-9]", "_", "/".join(parts)).upper()


def is_code(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and not stripped.startswith(("//", "/*", "*"))


def expected_lines(guard: str) -> tuple[str, str, str]:
    return f"#ifndef {guard}\n", f"#define {guard}\n", f"#endif // {guard}\n"


def check(path: str, fix: bool) -> list[str]:
    guard = guard_name(path)
    ifndef, define, endif = expected_lines(guard)
    lines = Path(path).read_text(encoding="utf-8").splitlines(keepends=True)
    first = next((i for i, line in enumerate(lines) if is_code(line)), None)
    last = next((i for i in range(len(lines) - 1, -1, -1) if is_code(lines[i])), None)
    problems: list[str] = []
    if first is None:
        return [f"{path}: empty header"]

    directive = DIRECTIVE.match(lines[first])
    kind = directive.group(1) if directive else None
    if kind == "pragma" and directive.group(2) == "once":
        problems.append(f"{path}: uses #pragma once instead of the include guard {guard}")
        if fix:
            lines[first:first + 1] = [ifndef, define]
            lines.append("\n" if lines[-1].endswith("\n") else "\n\n")
            lines.append(endif)
    elif kind == "ifndef" and first + 1 < len(lines) and DIRECTIVE.match(lines[first + 1]) and \
            DIRECTIVE.match(lines[first + 1]).group(1) == "define" and \
            last is not None and DIRECTIVE.match(lines[last]) and DIRECTIVE.match(lines[last]).group(1) == "endif":
        if lines[first] != ifndef or lines[first + 1] != define or lines[last] != endif:
            problems.append(f"{path}: include guard is not named {guard}")
            if fix:
                lines[first], lines[first + 1], lines[last] = ifndef, define, endif
    else:
        problems.append(f"{path}: no include guard; expected {guard}")
        if fix:
            lines[first:first] = [ifndef, define, "\n"]
            if not lines[-1].endswith("\n"):
                lines[-1] += "\n"
            lines += ["\n", endif]

    if fix and problems:
        Path(path).write_text("".join(lines), encoding="utf-8")
    return problems


def main(argv: list[str]) -> int:
    fix = "--fix" in argv
    files = [a for a in argv if a != "--fix"]
    if not files:
        files = subprocess.run(["git", "ls-files", "*.hpp"], capture_output=True, text=True,
                               check=True).stdout.split()
    problems = [p for f in files if f.endswith(".hpp") for p in check(f, fix)]
    for problem in problems:
        print(problem)
    if problems and fix:
        print(f"fixed {len(problems)} header(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
