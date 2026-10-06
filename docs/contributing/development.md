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
