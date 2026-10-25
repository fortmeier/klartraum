# Profiling

Use profiling to find expensive GPU work and compare shader implementations.

Under the hood, profiling inserts GPU timestamp commands before and after each
graph element's recorded work. The GPU writes its clock values into Vulkan query
pools; after execution finishes, the CPU reads them, converts the differences
to milliseconds and averages them across executions. Optional performance
counters use Vulkan queries to collect hardware measurements or pipeline
statistics, such as the number of compute-shader invocations.

Choose the workflow that matches what you want to measure:

- [A. Profile a full graph](#a-profile-a-full-graph) to find bottlenecks in a pipeline.
- [B. Profile a single element](#b-profile-a-single-element) to optimize a shader.
- [C. Profile the frame loop](#c-profile-the-frame-loop) to measure the running application.

Use a Release build with validation disabled for performance measurements.
Keep inputs, resolution, backend and device settings the same between runs.

## A. Profile a full graph

1. Build the graph with the same elements and dependencies as the application.
   Call `graph.enableProfiling()` **before** `graph.compileFrom(output)`, where
   `output` is the final element of the pipeline.
2. Initialize its inputs and run several warm-up executions to let caches and
   GPU clock state settle. Call `graph.resetProfiling()` to discard those timings.
3. Run repeated executions with representative inputs, then read the results.
   The function below starts with ten warm-up and fifty measured executions;
   increase these counts if repeated runs vary too much.

This function takes an already compiled graph with profiling enabled and its
inputs initialized. It submits path 0 and prints the mean GPU times:

```cpp
#include <iostream>
#include "klartraum/computegraph/computegraph.hpp"

void profileGraph(klartraum::VulkanContext& vc, klartraum::ComputeGraph& graph) {
    for (int i = 0; i < 10; ++i) {
        graph.submitAndWait(vc.getGraphicsQueue(), 0);
    }
    graph.resetProfiling();
    for (int i = 0; i < 50; ++i) {
        graph.submitAndWait(vc.getGraphicsQueue(), 0);
    }

    std::cout << "Graph GPU span: " << graph.getGpuSpanMs() << " ms\n";
    const auto times = graph.getElementTimesMs();
    const auto& elements = graph.getElements();
    for (size_t i = 0; i < elements.size(); ++i) {
        const char* name = elements[i]->getName();
        std::cout << (name[0] ? name : elements[i]->getType())
                  << ": " << times[i] << " ms\n";
    }
}
```

`submitAndWait()` collects the results automatically. `resetProfiling()` clears
the means and discards unread results; repeated reads without new submissions
leave the means unchanged.

### Reading the results

| Result | Use it for |
|---|---|
| `getGpuSpanMs()` | Mean GPU time for the whole graph, including barriers between elements. |
| `getElementTimesMs()` | Mean time per element, in `getElements()` order; use this to find expensive elements. |
| `getProfilingResults()` | Named means for printing, plus any enabled performance-counter results. |

The graph span can exceed the sum of element times because barriers and gaps
fall outside the individual measurements. Buffer and grouping elements also
receive timings, even when they record no GPU work. CPU work, loading and
presentation are outside these measurements; use workflow C for frame throughput.

## B. Profile a single element

For shader optimization, build a small graph containing the target element
and the resources it needs. Use representative data and dispatch dimensions.
Enable profiling before compiling, then use the warm-up and measurement routine
from workflow A. Select the target's own timing from `getElementTimesMs()`.

For example, with an initialized `VulkanContext`, this function measures an
element-wise multiplication. It calls `profileGraph()` from above:

```cpp
#include <memory>
#include <vector>
#include "klartraum/computegraph/bufferelement.hpp"
#include "klartraum/computegraph/generalcomputation.hpp"

void profileMultiply(klartraum::VulkanContext& vc) {
    using namespace klartraum;
    using Buffer = BufferElement<VulkanBuffer<float>>;
    constexpr uint32_t count = 16384;

    auto input = std::make_shared<Buffer>(vc, count);
    auto factor = std::make_shared<Buffer>(vc, count);
    auto output = std::make_shared<Buffer>(vc, count);
    auto multiply = std::make_shared<GeneralComputation<>>(
        vc, "shaders/operator_multiply_scalar_element_wise.comp.spv");
    multiply->setName("Multiply");
    multiply->setInput(input, 0);
    multiply->setInput(factor, 1);
    multiply->setInput(output, 2);
    multiply->setGroupCountX(count);

    ComputeGraph graph(vc, 1);
    graph.enableProfiling();
    graph.compileFrom(multiply);
    input->getBuffer(0).memcopyFrom(std::vector<float>(count, 3.0f));
    factor->getBuffer(0).memcopyFrom(std::vector<float>(count, 2.0f));
    profileGraph(vc, graph);

    const auto times = graph.getElementTimesMs();
    const auto& elements = graph.getElements();
    for (size_t i = 0; i < elements.size(); ++i) {
        if (elements[i] == multiply) {
            std::cout << "Target element: " << times[i] << " ms\n";
        }
    }
}
```

After changing the shader, rebuild its SPIR-V and recreate the element and
compiled graph. Compare repeated runs with the same inputs, and check output
correctness separately.

The selected timing covers all GPU commands recorded by that element. To
measure one shader, use an element that records just its dispatch. Profiling
still instruments every element in the small graph; there is no per-element
enable switch.

A small graph changes cache state and resource contention. Verify a promising
optimization in the full graph, then check application performance with profiling
disabled. More samples reduce noise but do not remove profiling overhead.

## C. Profile the frame loop

Use GPU timings to locate expensive work, and measure frame throughput to see
its effect on the application.

1. Call `engine.enableProfiling()` before adding graphs or registering a graph
   builder.
2. Run a fixed number of frames with a repeatable scene, camera path and rendering
   settings.
3. Call `engine.getProfilingResults()` to inspect named element times. The engine
   collects timestamps after each frame, and repeated reads do not count them
   again. Its public API has no profiling reset, so these averages include
   startup frames. Use workflow A for graph timings with a warm-up/reset boundary.

The Gaussian splatting example enables timestamps and performance counters with
`--frames` and prints the results after the loop ends. From the repository root,
for a single-configuration build:

```bash
./build/examples/gaussian_splatting_example --backend raster --frames 120
```

For a Visual Studio Release build:

```powershell
.\build\examples\Release\gaussian_splatting_example.exe --backend raster --frames 120
```

### Measure frame throughput

GPU element times do not include CPU work or presentation, and their sum is
not the frame time. To measure throughput, run your frame loop with profiling
disabled and time many frames after warm-up, excluding loading and setup:

- Mean frame time = elapsed milliseconds / frame count.
- FPS = frame count / elapsed seconds.

Keep presentation and VSync settings fixed; a presentation cap can hide GPU
improvements. Engine profiling waits for the GPU each frame, reducing CPU/GPU
overlap. `engine.disableProfiling()` stops that wait and collection, but already
compiled graphs retain their timestamp commands. Rebuild them with profiling
disabled for a comparison without instrumentation.

## Read results after a graph chain

When using `GraphChain`, enable profiling before compiling each graph, then
call `readProfiling()` on each graph after the chain has finished. For example,
with a compiled graph that has no external semaphore waits:

```cpp
#include "klartraum/computegraph/graphchain.hpp"

void profileInChain(klartraum::VulkanContext& vc, klartraum::ComputeGraph& graph) {
    klartraum::GraphChain chain(vc);
    chain.add(graph, 0);
    for (int i = 0; i < 10; ++i) {
        chain.submitAndWait(vc.getGraphicsQueue());
        graph.readProfiling();
    }
}
```

Query slots are shared across a graph's paths and executions. Finish and collect
each execution before another overwrites its results. If the same graph appears
several times in one chain, only its final occurrence's results remain. Measure
the chain separately if you need its complete duration.

## Export a graph

Include `klartraum/computegraph/graph_export.hpp` and call
`writeGraphJson("graph.json", "My graph", graph, {})` after collecting results.
The JSON contains elements, dependency edges and mean GPU times when available.
The last argument optionally defines named groups for visualization; see the
{doc}`API reference <../api/compute-graph>` for `ExportGroup`.

Write snapshots outside the measured frame loop to keep file I/O out of the
comparison.

## Performance impact

Profiling adds timestamps around every element and reads results back to the
CPU. This can be a substantial cost for short dispatches. For example, a Release
build on an RTX 2070 SUPER, with validation disabled, gave these GPU times:

| Workload | Profiling off | Profiling on |
|---|---|---|
| Lantern, raster backend | 1.51 ms | 1.55 ms |
| Lantern, compute backend | 26.85 ms | 26.85 ms |
| 64 small dispatches | 0.38 ms | 0.83 ms |

The lantern used 145,562 Gaussians at 512 by 512 pixels. Each value is the median
of three run medians, with twelve warm-up and sixty measured executions per
variant. Both variants waited for each execution, so this comparison excludes
the additional throughput cost of making an asynchronous frame loop wait.
The compute result showed no consistent difference across runs.

Use profiled timings to guide optimization, then confirm improvements with
profiling disabled and the graphs rebuilt without timestamp commands.

## Performance counters

For additional information such as compute-shader invocation counts, call
`enablePerformanceProfiling()` before compilation. It uses hardware counters
when available, with pipeline statistics as a fallback. An optional name filter,
such as `{"SM"}`, selects hardware counters. Call `enableProfiling()` separately
if you also want timestamp timings.

Counter results follow the timestamp entries in `getProfilingResults()`. They
have counter-specific units, so do not interpret every returned value as
milliseconds. Availability and collection costs depend on the device and driver.

See {doc}`../concepts/compute-graph` for graph execution and
{doc}`../api/compute-graph` for the API.
