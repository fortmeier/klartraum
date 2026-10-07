# Development setup

## Checks before each commit

The repository uses [pre-commit](https://pre-commit.com/) to check every commit,
for example for large files that belong in Git LFS or for leftover merge-conflict
markers. The checks are listed in `.pre-commit-config.yaml` in the repository
root.

Install pre-commit once, for example with [uv](https://docs.astral.sh/uv/),
and enable it in each clone:

```bash
uv tool install pre-commit
pre-commit install
```

The Git hook calls the installed `pre-commit`, so use `uv tool install` rather
than `uvx`: a hook set up through `uvx` points into uv's cache and stops working
when the cache is cleaned.

From then on the checks run on the staged files whenever you commit. If a check
fails or changes a file, the commit is stopped; review the result, stage it and
commit again. To run all checks on the whole repository:

```bash
pre-commit run --all-files
```

## Copyright and license information

The repository follows the [REUSE](https://reuse.software/) specification:
every file states its copyright holder and license. Source files carry an SPDX
header at the top:

```cpp
// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT
```

Documentation, configuration, lock files, the website and data are covered by
`REUSE.toml` in the repository root instead. The license texts are in
`LICENSES/`. The pre-commit check rejects committed files without this
information. To add the header to new source files, run:

```bash
scripts/dev/reuse_annotate.sh
```

To check the whole repository:

```bash
uvx --from 'reuse[charset-normalizer]' reuse lint
```
