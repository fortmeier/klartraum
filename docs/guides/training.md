# Training

Train networks on the GPU with the same pre-recorded compute graphs that run
them. A training step is a graph with three parts:

1. The forward elements compute the network's output.
2. The backward elements compute every gradient.
3. The optimiser elements update the parameters.

The graph is recorded once and submitted once per step. No framework runs
alongside it; PyTorch only provides the reference values of the tests.

- [A. Building blocks](#a-building-blocks) lists the elements.
- [B. A training step](#b-a-training-step) shows how they fit together.
- [C. Testing an element](#c-testing-an-element) compares it with PyTorch.

## A. Building blocks

The elements are factories in `klartraum::training` (`layers/training.hpp`).
Each returns a compute graph element whose slots take the tensors. The forward
layers are those of `klartraum::layers`, the same ones that run ONNX networks.

| Group | Elements |
|---|---|
| Plumbing | `fill`, `sum`, `channelCopy` (concatenate, split, accumulate), `affine` |
| Activations | `activationBackward` (ReLU, sigmoid, ELU, leaky ReLU), `leakyRelu` |
| Dense layers | `linear`, `linearBackwardInput`, `linearBackwardWeights`, `columnSum` |
| Convolutions | `convBackwardInput`, `convBackwardWeights` (split-K, register-blocked) |
| Normalization | `channelNorm` and its backward passes; `film`, `filmBackward` |
| Resampling | `upsample2xBackward`, `maxPool`, `averagePool` and their backward passes |
| Tables | `embedding`, `embeddingBackward` (with an L2 penalty) |
| Optimiser | `adamStep` (exponential or cosine decay), `stepCounter`, `squaredNorm`, `clipGradient`, `ema` |
| Losses | `photometricLoss`, `l1Distance`, `neural::VggLoss`, `softplusLoss`, `flowWarpLoss` |
| GAN | `augmentPair`, `augmentPairBackward` (DiffAugment with given random numbers) |
| Latents | `objectMean`, `objectBroadcast`, `vae`, `vaeBackward` (KL with free bits) |
| Textures | `neural::hashTexture` and its backward pass, `positionalEncoding`, `textureBasis` and its backward pass and regularisation |

Simple activation gradients come from Slang's automatic differentiation
(`shaders/training/differentiable.slang`). `MlpTraining` builds a whole training
step of a fully connected network from these elements; see
`autodiff_training_example`.

## B. A training step

A step is a sequence of elements, each depending on the one before. Add them in
order and make each depend on the previous one (`addDependency()`), so the graph
runs them one after another in a single command buffer. Then:

1. **Forward:** the layers, writing every intermediate tensor the backward pass
   needs (pre-activations, normalization statistics).
2. **Loss:** a loss element writes the output's gradient and each element's
   share of the loss. `sum()` of the shares gives the loss for logging.
3. **Backward:** the backward elements in reverse order. A gradient that several
   paths reach is added (`accumulate`, `channelCopy(..., true)`). Gradients
   scattered by atomic additions (`embeddingBackward`, the textures) need the
   gradient zeroed first with `fill()`.
4. **Optimiser:**
   - `squaredNorm()` of all gradients, then `clipGradient()` of each;
   - `stepCounter()` once;
   - `adamStep()` and `ema()` per parameter.

   Every Adam step reads the counter, so the bias correction stays right while
   the same command buffer is submitted again and again.

Parameter groups (for example network weights, a texture and per-image codes)
are simply separate `adamStep()` elements with their own `AdamParameters`.
Atomic float additions need `VulkanContext::isBufferFloatAtomicAddSupported()`.

## C. Testing an element

`tests/test_training_elements.cpp` compares each element with PyTorch:

1. `scripts/neural/make_element_fixtures.py` writes, for every case, random
   inputs, an upstream gradient, and PyTorch's outputs and autograd gradients
   to `build/TestingOutput/training_fixtures/`.
2. The PyTorch references of Klartraum's own layers (ChannelNorm, the VAE,
   DiffAugment, the textures) are in `scripts/neural/training_reference.py`.
3. The tests skip when a fixture is missing.

```bash
uv run --project scripts/qwen_image python scripts/neural/make_element_fixtures.py
uv run --project scripts/qwen_image python scripts/neural/export_vgg16.py   # for the VGG loss
./build/klartraum_tests --gtest_filter='TrainingElements.*'
```

To add an element:

1. Write its shader in `shaders/training/` and its factory in
   `layers/training.hpp`.
2. Add a case to `make_element_fixtures.py` and a test that runs the element on
   the fixture's tensors.
3. Compare with `relativeError()` (`tests/training_fixtures.hpp`), relative to
   the reference's largest magnitude.
