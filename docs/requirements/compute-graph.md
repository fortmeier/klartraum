# Compute graph

```{req} Compiled before use
:id: REQ_GRAPH_PRERECORDED
:status: implemented

A compute graph shall be defined and compiled fully into command buffers
before it can be submitted. Submitting a compiled graph shall not record
command buffers.
```

```{req} Multiple execution paths
:id: REQ_GRAPH_PATHS
:status: implemented

A compiled graph shall support a fixed number of execution paths, each with
its own command buffers, so that several frames can be in flight at once.
```

```{req} Parallel execution on several queues
:id: REQ_GRAPH_PARALLEL_QUEUES
:status: open

Compute graphs shall be able to run in parallel when more than one queue that
can run compute graphs is available.
```

```{req} Use from several threads
:id: REQ_GRAPH_THREADS
:status: implemented

Compute graphs shall be creatable and submittable from different threads for
the same Vulkan context.
```

```{req} Outputs shared across queues
:id: REQ_GRAPH_SHARED_OUTPUTS
:status: implemented

The output of a compute graph shall be usable as input by a compute graph
running on another queue.
```

```{spec} Topological ordering with Kahn's algorithm
:id: SPEC_GRAPH_ORDER
:status: implemented
:implements: REQ_GRAPH_PRERECORDED

`ComputeGraph::compileFrom()` orders the elements with Kahn's algorithm, sets
them up and records one command buffer per path.
```

```{test} Headless submit and wait
:id: TEST_GRAPH_HEADLESS_SUBMIT
:status: verified
:verifies: SPEC_GRAPH_ORDER, REQ_GRAPH_PATHS
:gtest: ComputeGraph.headlessSubmitAndWait

Compiles a render graph over several headless paths and submits each path
with `submitAndWait`.
```

```{test} Background graph beside the frames
:id: TEST_GRAPH_BACKGROUND_BESIDE_FRAMES
:status: verified
:verifies: REQ_GRAPH_THREADS, REQ_GRAPH_SHARED_OUTPUTS
:gtest: ComputeGraph.backgroundQueueRunsBesideFrames

Builds and runs a graph on a worker thread while the main thread renders
frames, then reads the graph's output on the graphics queue.
```
