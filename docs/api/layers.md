# Neural-network layers

The layers {cpp:class}`klartraum::OnnxNetwork` executes, as factories that
take plain shapes and attributes. Each returns one compute-graph element; an
application can place layers directly in a compute graph, and `OnnxNetwork`
builds its operations with the same factories after reading shapes and
attributes from the model.

Connect a layer's input tensors with `setInput(tensor, slot)` at slots 0, 1,
... in the order its factory lists them, then its output tensors at the
following slots. Scaling latents by a constant, for example:

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
