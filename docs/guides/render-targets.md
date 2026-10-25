# Render targets and image composition

An image can be the background for later drawing in a compute graph. For
example, first clear a target, then place a background image on it, then draw
Gaussians over the result. The graph orders these steps through their inputs.

## Clearing and keeping contents

An `ImageViewSrc` clears its image to opaque black each time its path runs.
`OffscreenTarget` owns the images and also initializes them to opaque black
when it is constructed. Call `setClearColor()` to change the per-execution
clear, or `setClear(false)` to keep the contents instead.

`ClearImage` is an explicit clear in the graph. It exposes the image it clears,
so the next element can use it as its target. When using it, disable the
underlying source's clear to avoid clearing twice.

Set clearing, placement and render-pass loading options **before
`compileFrom()`**. They affect recorded commands; changing them afterwards
does not change the compiled graph. A plain `RenderPass` clears its attachment
unless configured with `setLoadExisting(true)`; both Gaussian backends draw
over their target's current contents.

## Place an image on a target

`ImageComposite` reads a source and writes a distinct target. Set its source
first, then its target. It exposes that target after the draw, so renderers
connected to the composite run after it.

| Option | Result |
|---|---|
| `CompositeMode::Replace` | Copy the filtered source RGBA into the drawn area. |
| `CompositeMode::Over` | Blend the filtered source over the target using source alpha. |
| `CompositeFit::Stretch` | Cover the target, changing the aspect ratio if necessary. |
| `CompositeFit::Fit` | Centre the whole source with its aspect ratio preserved; keep the target in the margins. |
| `CompositeFit::Fill` | Centre and crop the source to cover the target with its aspect ratio preserved. |

The source is sampled bilinearly, with reads clamped to its edges. Both images
need a format compatible with the shader's `rgba8` storage images, and their
actual extents must match the extents supplied to the composite.

## Example: background, then Gaussians

This function renders one offscreen image using an initialized `VulkanContext`,
a Gaussian scene and a camera whose matrices have been set for a square
target. Its background is a tall, opaque blue image fitted onto black; either
Gaussian backend can draw over it. The returned target remains available
after the graph has finished.

```cpp
#include <memory>
#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/imagecomposite.hpp"
#include "klartraum/gaussian_splatting_factory.hpp"
#include "klartraum/offscreen_target.hpp"

std::shared_ptr<klartraum::OffscreenTarget> renderComposedImage(
    klartraum::VulkanContext& vc,
    std::shared_ptr<klartraum::GaussianDataStandard> scene,
    std::shared_ptr<klartraum::CameraUboType> camera,
    klartraum::GsplatBackend backend) {
    using namespace klartraum;
    const VkExtent2D backgroundExtent{64, 128}, extent{256, 256};

    auto background = std::make_shared<OffscreenTarget>(vc, backgroundExtent, 1);
    background->setClearColor({{0.1f, 0.2f, 0.5f, 1.0f}});

    auto target = std::make_shared<OffscreenTarget>(vc, extent, 1);
    target->setClear(false);
    auto clear = std::make_shared<ClearImage>(); // opaque black
    clear->setInput(target);

    auto composite = std::make_shared<ImageComposite>(
        vc, backgroundExtent, extent, CompositeMode::Replace, CompositeFit::Fit);
    composite->setSource(background);
    composite->setTarget(clear);

    auto splats = createGaussianSplatting(vc, backend, composite, camera, scene);
    ComputeGraph graph(vc, 1);
    graph.compileFrom(splats);
    camera->update(0);
    graph.submitAndWait(vc.getGraphicsQueue(), 0);
    return target;
}
```

The target source keeps its contents; `ClearImage` establishes the black
background, and the composite places the blue image onto it. Connecting the
Gaussian renderer to the composite makes all of those steps its dependencies.
The renderer finishes this offscreen target in `TRANSFER_SRC_OPTIMAL`, ready
for a copy or blit. Keep the target and its Vulkan context alive until all
later users of that image have finished.

## Paths and one shared image

An `OffscreenTarget` normally owns one image per path. Allocate as many images
as the graph has paths, so frames in flight do not overwrite each other.

`SinglePathImage` provides one image for every path. Use it for a result all
paths read, such as a background computed separately. Its graph readers must
only read it. `copyFrom()` replaces its contents synchronously, requires a
source with the same extent and format, and restores the source's layout.
Wait for the source computation and all previous readers before copying.
Pass the background queue explicitly if the copy should run there.

`ImageViewForward` passes an image through while retaining a dependency on its
producer. For example, if an operation writes an image through input slot 1,
use `forward->setInput(operation, 0, 1)` to expose that image after the write.
`ImageComposite` and `ClearImage` already provide this forwarding themselves.

## Layouts when keeping contents

An uncleared source must already be in `GENERAL` when the graph uses it.
Clearing transitions to `GENERAL`. The compositor also expects its target in
`GENERAL`; its `sourceLayout` argument specifies the source's incoming layout,
which it transitions to `GENERAL` and leaves there.

The Gaussian backends leave images in their target's final layout. For an
offscreen target this is `TRANSFER_SRC_OPTIMAL`; for a swapchain image it is
normally `PRESENT_SRC_KHR`. Before rendering over that image again with
clearing disabled, a transition back to `GENERAL` must preserve its contents.
That transition is not added automatically by an uncleared image source.
Automatic handling across repeated renders is tracked in
[#64](https://github.com/fortmeier/klartraum/issues/64). The example above
renders once and does not rely on that behavior.

## Alpha conventions

In straight-alpha RGBA, RGB contains the full color and alpha is separate.
In premultiplied RGBA, RGB has already been multiplied by alpha. Half-opaque
red is `(1, 0, 0, 0.5)` in the first representation and `(0.5, 0, 0, 0.5)`
in the second.

The current compositor's `Over` mode treats its source as straight alpha:

```text
rgb = source.rgb * source.a + target.rgb * (1 - source.a)
a   = source.a + target.a * (1 - source.a)
```

Gaussian renderers accumulate premultiplied RGB. Feeding their transparent
output into this formula applies the source opacity again. `Replace` copies
filtered RGBA without converting its representation, and filtering a
straight-alpha source does not account for premultiplied contributions at
transparent edges. A consistent convention and transparent-layer tests are
tracked in [#63](https://github.com/fortmeier/klartraum/issues/63). The example
uses opaque backgrounds and does not composite transparent Gaussian output.

See {doc}`../requirements/render-targets` for requirements, specifications and
the linked rendering tests, and {doc}`../api/compute-graph` for the API.
