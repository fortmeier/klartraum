# Neural-network layers

In Klartraum, neural networks are composed by layer compute-graph elements.
To instantiate a neural network with structure and parameters as given by
an .onnx file, the class {cpp:class}`klartraum::OnnxNetwork` can be used.
This class uses special layer factories.
Each returns one compute-graph element; an
application can place layers directly in a compute graph, and `OnnxNetwork`
builds its operations with the same factories after reading shapes and
attributes from the model.

Connect a layer's input tensors with `setInput(tensor, slot)` at slots 0, 1,
... in the order its factory lists them, then its output tensors at the
following slots.

Scaling latents by a constant, for example:

```cpp
using namespace klartraum;
const layers::Shape shape{1, 4, 64, 64};
auto scale = layers::binary(vulkanContext, layers::BinaryOp::Mul, shape, {1}, shape);
scale->setInput(latents, 0);   // 1x4x64x64
scale->setInput(factor, 1);    // one element
scale->setInput(scaled, 2);    // output, 1x4x64x64
```

```{doxygennamespace} klartraum::layers
:members:
```
