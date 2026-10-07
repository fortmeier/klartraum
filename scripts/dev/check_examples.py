# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

"""Checks that examples/README.md documents every example and only real options.

Each examples/<name>.cpp needs a section headed "### `<name>`" in the README,
and every command-line option (--something) named in that section has to appear
as the string literal "--something" in examples/<name>.cpp. Exits with 1 and
lists the problems otherwise. Run from anywhere; used as a pre-commit hook.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPOSITORY_ROOT / "examples"
README = EXAMPLES_DIR / "README.md"

SECTION_HEADING = re.compile(r"^### `([A-Za-z0-9_]+)`\s*$")
OPTION = re.compile(r"(?<![\w-])--[a-z][a-z0-9-]*")


def readme_sections(text: str) -> dict[str, str]:
    """Maps each example name to the text of its README section."""
    sections: dict[str, str] = {}
    name = None
    lines: list[str] = []
    for line in text.splitlines():
        match = SECTION_HEADING.match(line)
        if match or line.startswith("## ") or line.startswith("# "):
            if name is not None:
                sections[name] = "\n".join(lines)
            name = match.group(1) if match else None
            lines = []
        elif name is not None:
            lines.append(line)
    if name is not None:
        sections[name] = "\n".join(lines)
    return sections


def main() -> int:
    examples = sorted(path.stem for path in EXAMPLES_DIR.glob("*.cpp"))
    sections = readme_sections(README.read_text(encoding="utf-8"))
    problems = []

    for name in examples:
        if name not in sections:
            problems.append(f"{name}: no section '### `{name}`' in examples/README.md")
    for name in sorted(sections):
        if name not in examples:
            problems.append(f"{name}: documented in examples/README.md, but examples/{name}.cpp does not exist")
            continue
        source = (EXAMPLES_DIR / f"{name}.cpp").read_text(encoding="utf-8")
        for option in sorted(set(OPTION.findall(sections[name]))):
            if f'"{option}"' not in source:
                problems.append(f"{name}: README names {option}, which examples/{name}.cpp does not parse")

    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
