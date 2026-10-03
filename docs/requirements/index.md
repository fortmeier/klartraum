# Requirements

Requirements, specifications and tests are tracked with
[sphinx-needs](https://sphinx-needs.readthedocs.io/). Each item has an ID, a
status and links to the items it refines, implements or verifies.

| Type | Prefix | Links |
|---|---|---|
| Requirement | `REQ_` | |
| Specification | `SPEC_` | `implements` a requirement |
| Test | `TEST_` | `verifies` a specification or requirement; `gtest` names the GoogleTest case |

Statuses: `draft`, `open`, `implemented`, `verified`.

```{note}
This section is at an early stage. The items below are a starting set that
shows the structure; they are not a complete specification of the engine.
```

## Overview

```{needtable}
:columns: id, title, type, status, implements, verifies
:style: table
```

```{toctree}
:maxdepth: 1

compute-graph
```

## Gaussian splatting

```{req} Interchangeable splatting backends
:id: REQ_GSPLAT_BACKENDS
:status: implemented

The compute and the raster Gaussian splatting backends shall take the same
inputs, so that switching the backend does not change the surrounding graph.
```

```{test} Both backends agree on the raccoon scene
:id: TEST_GSPLAT_BACKENDS_AGREE
:status: verified
:verifies: REQ_GSPLAT_BACKENDS
:gtest: GaussianSplattingFactory.bothBackendsAgreeOnRaccoonScene

Renders the raccoon scene with both backends and compares the images.
```
