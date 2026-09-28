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

## Compute graph

```{req} Pre-recorded GPU work
:id: REQ_GRAPH_PRERECORDED
:status: implemented

Running a compiled graph shall not record Vulkan command buffers; it shall
only submit command buffers recorded at compile time.
```

```{req} Multiple execution paths
:id: REQ_GRAPH_PATHS
:status: implemented

A compiled graph shall support a fixed number of execution paths, each with
its own command buffers, so that several frames can be in flight at once.
```

```{spec} Topological ordering with Kahn's algorithm
:id: SPEC_GRAPH_ORDER
:status: implemented
:implements: REQ_GRAPH_PRERECORDED

`ComputeGraph::compileFrom()` orders the elements with Kahn's algorithm, sets
them up and records one command buffer per element and path.
```

```{test} Headless submit and wait
:id: TEST_GRAPH_HEADLESS_SUBMIT
:status: verified
:verifies: SPEC_GRAPH_ORDER, REQ_GRAPH_PATHS
:gtest: ComputeGraph.headlessSubmitAndWait

Compiles a render graph over several headless paths and submits each path
with `submitAndWait`.
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
