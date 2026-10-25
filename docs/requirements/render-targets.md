# Render targets and image composition

For usage and examples, see {doc}`../guides/render-targets`.

## Requirements

```{req} Default frame contents
:id: REQ_IMAGE_DEFAULT_CLEAR
:status: implemented

An image source shall clear its image to opaque black on each execution by
default, so drawing starts with defined contents.
```

```{req} Configurable clear color
:id: REQ_IMAGE_CLEAR_COLOR
:status: implemented

Applications shall be able to select the color used to clear a target.
```

```{req} Preserve existing contents
:id: REQ_IMAGE_PRESERVE
:status: implemented

Applications shall be able to preserve a target's existing contents as the
background for later drawing in the graph.
```

```{req} Gaussian rendering over a background
:id: REQ_IMAGE_GSPLAT_OVER
:status: implemented

Both Gaussian splatting backends shall draw over a target's existing
background, preserving uncovered pixels and blending covered pixels with it.
```

```{req} Image composition and placement
:id: REQ_IMAGE_COMPOSITE
:status: implemented

Applications shall be able to replace target pixels with a source image or
blend it over them using source alpha. Placement shall support stretching,
fitting the whole source with its aspect ratio preserved, and filling the
target by cropping the source with its aspect ratio preserved. Pixels outside
the placed source shall retain the target's contents.
```

```{req} One image shared by execution paths
:id: REQ_IMAGE_SHARED_PATHS
:status: implemented

Several execution paths shall be able to read the same image without
modifying it. After all readers have finished, applications shall be able to
replace its contents for subsequent executions without recompiling the graph.
```

```{req} Consistent transparent image composition
:id: REQ_IMAGE_ALPHA_CONSISTENCY
:status: open

Transparent images produced by either Gaussian backend shall be composable
over another image without applying opacity twice. Resizing shall preserve
the contributions of colors at transparent edges.

The existing straight-alpha composition test does not verify this requirement.
```

```{req} Repeated rendering with preserved contents
:id: REQ_IMAGE_PRESERVE_REUSE
:status: open

A compiled rendering graph shall support repeated execution on the same
offscreen target with clearing disabled, preserving its contents and handling
the layout required by the next execution.

The existing preservation test submits an image source without a Gaussian
renderer.
```

## Specifications

```{spec} Image clearing
:id: SPEC_IMAGE_CLEAR
:status: implemented
:implements: REQ_IMAGE_DEFAULT_CLEAR, REQ_IMAGE_CLEAR_COLOR, REQ_IMAGE_PRESERVE

`ImageViewSrc` records a clear to opaque black unless `setClear(false)` was
called before compilation. `setClearColor()` selects its clear color.
`ClearImage` records an explicit clear of its input and forwards that image.
Clearing leaves the image in `GENERAL`; disabling it requires the image to be
in `GENERAL` already. Images must support transfer destination usage.
`OffscreenTarget` initializes its images with an opaque black clear in its
constructor.
```

```{spec} Render over the target
:id: SPEC_IMAGE_GSPLAT_OVER
:status: implemented
:implements: REQ_IMAGE_GSPLAT_OVER, REQ_IMAGE_PRESERVE, REQ_GSPLAT_BACKENDS

The raster backend uses `RenderPass::setLoadExisting(true)` to load the
attachment. The compute backend reads the background and adds its RGB scaled
by the Gaussians' remaining transmittance to their accumulated RGB. Both
expect the target in `GENERAL` and leave it in the target's final layout.
The tests below cover opaque backgrounds. Transparent image composition and
automatic layout handling across repeated uncleared renders remain open.
```

```{spec} Composite an image
:id: SPEC_IMAGE_COMPOSITE
:status: implemented
:implements: REQ_IMAGE_COMPOSITE

`ImageComposite` reads a source and writes a distinct target in place. Both
must support the shader's `rgba8` storage-image format. Extents are supplied
when the element is constructed. The source is transitioned from the supplied
layout to `GENERAL` and remains there; the target must already be in
`GENERAL` and remains there too.

`Replace` writes the bilinearly filtered source RGBA. `Over` uses the filtered
source's straight RGB and alpha: `rgb = source.rgb * source.a + target.rgb *
(1 - source.a)` and `a = source.a + target.a * (1 - source.a)`. This does not
convert premultiplied sources. Stretch covers the target; Fit and Fill
preserve the aspect ratio and centre the source. Sampling clamps to the
source edges, and pixels outside a fitted source are left unchanged.
```

```{spec} Forward the image and its dependency
:id: SPEC_IMAGE_FORWARD
:status: implemented
:implements: REQ_IMAGE_COMPOSITE, REQ_IMAGE_GSPLAT_OVER

`ImageViewForward` exposes the image, view, extent and final-layout override
of input 0, including an input selected through an output slot. It does not
clear the image. Connecting through it makes the producer a graph dependency.
`ClearImage` and `ImageComposite` use this interface so later renderers can
draw onto their result.
```

```{spec} Shared read-only image
:id: SPEC_IMAGE_SHARED_PATHS
:status: implemented
:implements: REQ_IMAGE_SHARED_PATHS

`SinglePathImage` owns one image and returns it for every path. It is not
cleared by the graph and must only be read by graph elements. `copyFrom()`
synchronously replaces its contents from an image with the same extent and
format, restores the source layout, and leaves the shared image in `GENERAL`.
The caller must wait for previous readers and the source producer to finish.
```

## Tests

```{test} Default opaque black
:id: TEST_IMAGE_DEFAULT_CLEAR
:status: verified
:verifies: SPEC_IMAGE_CLEAR, REQ_IMAGE_DEFAULT_CLEAR
:gtest: RenderOverTargetsTest.offscreenTargetClearsToBlackByDefault

Fills an image with white, then runs its source and checks every pixel is
opaque black.
```

```{test} Configurable clear color
:id: TEST_IMAGE_CLEAR_COLOR
:status: verified
:verifies: SPEC_IMAGE_CLEAR, SPEC_IMAGE_FORWARD, REQ_IMAGE_CLEAR_COLOR
:gtest: RenderOverTargetsTest.clearImageUsesItsColor

Clears an uncleared source through `ClearImage` and checks its selected color
and forwarded image resources.
```

```{test} Preserve an image source's contents
:id: TEST_IMAGE_PRESERVE
:status: verified
:verifies: SPEC_IMAGE_CLEAR, REQ_IMAGE_PRESERVE
:gtest: RenderOverTargetsTest.clearCanBeDisabled

Fills a target with white and checks that submitting its image source with
clearing disabled preserves it. This does not exercise repeated rendering.
```

```{test} Gaussian backends over opaque backgrounds
:id: TEST_IMAGE_GSPLAT_OVER
:status: verified
:verifies: SPEC_IMAGE_GSPLAT_OVER, REQ_IMAGE_GSPLAT_OVER
:gtest: RenderOverTargetsTest.backendsRenderOverTheirTarget

Renders Gaussians over black and blue with both backends, checks uncovered
pixels and background contributions, and compares the resulting images.
```

```{test} Composition modes
:id: TEST_IMAGE_COMPOSITE
:status: verified
:verifies: SPEC_IMAGE_COMPOSITE, REQ_IMAGE_COMPOSITE
:gtest: RenderOverTargetsTest.compositeDrawsOntoTarget

Checks replacement, fitting, filling and a straight-alpha source blended over
an opaque background. This does not cover transparent Gaussian output.
```

```{test} Fit preserves the source pattern
:id: TEST_IMAGE_FIT_PATTERN
:status: verified
:verifies: SPEC_IMAGE_COMPOSITE, REQ_IMAGE_COMPOSITE
:gtest: RenderOverTargetsTest.compositeFitsPatternWithoutDistortion

Fits a tall pattern into a square target, checking every source pixel and the
untouched margins.
```

```{test} Fill crops the centre of the source
:id: TEST_IMAGE_FILL_PATTERN
:status: verified
:verifies: SPEC_IMAGE_COMPOSITE, REQ_IMAGE_COMPOSITE
:gtest: RenderOverTargetsTest.compositeFillsByCroppingCenter

Fills a square target from a wide pattern, checking that both sides are
cropped equally and that the remaining pixels are in the correct positions.
```

```{test} Opaque bilinear interpolation
:id: TEST_IMAGE_FILTER_PATTERN
:status: verified
:verifies: SPEC_IMAGE_COMPOSITE
:gtest: RenderOverTargetsTest.compositeInterpolatesOpaquePattern

Upscales four distinct opaque colors and checks edge pixels and interpolated
interior values. Transparent-edge filtering remains part of the open alpha
requirement.
```

```{test} Shared image across paths and updates
:id: TEST_IMAGE_SHARED_PATHS
:status: verified
:verifies: SPEC_IMAGE_SHARED_PATHS, SPEC_IMAGE_FORWARD, REQ_IMAGE_SHARED_PATHS
:gtest: RenderOverTargetsTest.singlePathImageIsReadByEveryPath

Copies one shared image into three per-path targets through an image forward.
Replaces the shared contents between completed runs, checks all paths see
each update, and checks that the shared image itself is unchanged by readers.
```
