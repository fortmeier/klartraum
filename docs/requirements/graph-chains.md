# Chains of graphs

## Requirements

```{req} Ordered execution of compiled graphs
:id: REQ_GRAPH_CHAIN_ORDER
:status: implemented

The engine shall support an ordered chain of already compiled graphs. Each
link shall see the results of the preceding links, without CPU submissions
or waits between links.
```

```{req} Repeated graphs
:id: REQ_GRAPH_CHAIN_REPEAT
:status: implemented

A chain shall support executing the same graph path more than once. Every
occurrence shall execute, using the results of the preceding links.
```

```{req} Reusable chains
:id: REQ_GRAPH_CHAIN_REUSE
:status: implemented

A chain shall be reusable after its previous submission has finished, using
updated inputs without recompiling its graphs or recording new command
buffers.
```

```{req} Data shared between graphs
:id: REQ_GRAPH_CHAIN_SHARED_DATA
:status: implemented

Graphs in a chain shall be able to share buffers, so that a result produced
by one graph can be used by a later graph without copying it through the CPU.
Compiling a later graph shall preserve the shared buffers and their contents.
```

```{req} Separate paths in flight
:id: REQ_GRAPH_CHAIN_PATHS
:status: implemented

Chains using different paths of the same graphs shall be able to have
submissions in flight at the same time. Each path shall use its own per-path
buffers without overwriting another path's data.
```

## Specifications

```{spec} Submit the chain as one batch
:id: SPEC_GRAPH_CHAIN_SUBMIT
:status: implemented
:implements: REQ_GRAPH_CHAIN_ORDER, REQ_GRAPH_CHAIN_REUSE, REQ_GRAPH_CHAIN_PATHS, REQ_GRAPH_PRERECORDED

`GraphChain` stores a sequence of graph and path pairs. It submits their
recorded command buffers in one `vkQueueSubmit`, with a semaphore between
consecutive links and a fence for completion of the whole chain. `wait()`
waits for that fence. Submitting the same chain again waits for its previous
submission before reusing its synchronization objects.
```

```{spec} Simultaneous use for repeated graph paths
:id: SPEC_GRAPH_CHAIN_REPEAT
:status: implemented
:implements: REQ_GRAPH_CHAIN_REPEAT

A graph path that occurs more than once in a chain must have been compiled
after `enableSimultaneousUse()`, which records its command buffer with
`VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT`. `GraphChain::add()` rejects
a repeated graph path without this flag.
```

```{spec} Reuse shared buffer allocations
:id: SPEC_GRAPH_CHAIN_BUFFERS
:status: implemented
:implements: REQ_GRAPH_CHAIN_SHARED_DATA, REQ_GRAPH_CHAIN_PATHS

A `BufferElement` shared by several graphs is allocated by the first graph
that compiles it. Later graphs reuse those allocations and must have the
same number of paths. Each path keeps its own allocation, and a chain link
selects which path of its graph runs.
```

## Tests

```{test} One graph in a chain
:id: TEST_GRAPH_CHAIN_SINGLE_LINK
:status: verified
:verifies: SPEC_GRAPH_CHAIN_SUBMIT
:gtest: GraphChain.singleLinkMatchesSubmitAndWait

Runs a graph on its own and then as a one-link chain, checking that both
submissions apply the same computation to its buffer.
```

```{test} Repeated graph paths require simultaneous use
:id: TEST_GRAPH_CHAIN_REPEAT_GUARD
:status: verified
:verifies: SPEC_GRAPH_CHAIN_REPEAT
:gtest: GraphChain.repeatedGraphNeedsSimultaneousUse

Adding the same graph path twice throws unless the graph was compiled after
`enableSimultaneousUse()`.
```

```{test} Every occurrence of a graph executes
:id: TEST_GRAPH_CHAIN_REPEAT_EXECUTES
:status: verified
:verifies: SPEC_GRAPH_CHAIN_REPEAT, REQ_GRAPH_CHAIN_REPEAT
:gtest: GraphChain.repeatedGraphRunsEveryTime

Runs a graph that doubles a buffer four times in one chain and checks that
the result is sixteen times the input.
```

```{test} Links execute in chain order
:id: TEST_GRAPH_CHAIN_ORDER
:status: verified
:verifies: SPEC_GRAPH_CHAIN_SUBMIT, REQ_GRAPH_CHAIN_ORDER, REQ_GRAPH_CHAIN_SHARED_DATA
:gtest: GraphChain.linksRunInChainOrder

Chains graphs that double a shared buffer or add one to it. Changing their
order changes the result as expected.
```

```{test} Reuse a chain across frames
:id: TEST_GRAPH_CHAIN_REUSE
:status: verified
:verifies: SPEC_GRAPH_CHAIN_SUBMIT, REQ_GRAPH_CHAIN_REUSE
:gtest: GraphChain.chainRunsEveryFrame

Submits the same chain for several frames, replacing its input between
submissions, and checks the result of each frame.
```

```{test} Shared buffers are allocated once
:id: TEST_GRAPH_CHAIN_SHARED_BUFFERS
:status: verified
:verifies: SPEC_GRAPH_CHAIN_BUFFERS, REQ_GRAPH_CHAIN_SHARED_DATA
:gtest: GraphChain.sharedBufferElementAllocatedOnce

Compiles two graphs sharing a per-path `BufferElement` and checks that the
second graph preserves the first graph's buffer handles. Compiling another
graph with a different number of paths throws.
```

```{test} Paths in flight keep separate data
:id: TEST_GRAPH_CHAIN_PATHS
:status: verified
:verifies: SPEC_GRAPH_CHAIN_SUBMIT, SPEC_GRAPH_CHAIN_BUFFERS, REQ_GRAPH_CHAIN_PATHS
:gtest: GraphChain.pathsInFlightStaySeparate

Submits two chains over different paths of the same graphs before waiting
for either chain. Both paths compute the expected result from their own
input buffers.
```
