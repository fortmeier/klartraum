# ONNX networks

{cpp:class}`klartraum::OnnxNetwork` loads a model in the
[ONNX](https://onnx.ai/) format and turns it into compute graph elements, so a
network runs on the GPU like any other part of the graph.

```cpp
auto network = vulkanContext.create<klartraum::OnnxNetwork>("./data/onnx/simple_encoder.onnx");
network->printModelInfo();

klartraum::ComputeGraph graph(vulkanContext, 1);
graph.compileFrom(network);
```

## Supported operators

| ONNX operator | Notes |
|---|---|
| `Conv` | 2D convolution |
| `ConvTranspose` | 2D transposed convolution |
| `Relu` | |
| `Reshape` | |
| `Transpose` | |
| `Constant` | Used for weights and shapes |

Support for ONNX is at an early stage. Models with other operators cannot be
loaded yet.
