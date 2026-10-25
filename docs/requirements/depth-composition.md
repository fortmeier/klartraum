# Depth composition

For usage and limitations, see {doc}`../guides/gaussian-splatting`.

```{req} Optional shared depth
:id: REQ_DEPTH_SHARED
:status: implemented

Render passes and either Gaussian backend shall be composable in either order
with an optional shared per-path depth attachment. Without an attachment,
rendering shall retain its color-only behavior.
```

```{req} Independent depth controls
:id: REQ_DEPTH_CONTROLS
:status: implemented

Each rendering stage shall independently enable or disable depth testing and
depth writing. Testing shall reject contributions behind the existing surface;
writing shall update the depth for subsequent stages. Disabling writes shall
leave existing depth unchanged.
```

```{req} Selectable Gaussian surface estimate
:id: REQ_DEPTH_METHODS
:status: implemented

Both Gaussian backends shall support accumulated-opacity and nearest-opaque-splat
depth, with a configurable opacity threshold. Defaults shall preserve each
backend's existing convention. Depth export shall be optional and independent
of the shared attachment's testing and writing controls.
```

```{req} Depth lifetime and paths
:id: REQ_DEPTH_PATHS
:status: implemented

Each execution path shall have independent depth storage. A shared attachment
shall clear once per graph execution by default, with optional preservation
between executions. Mismatched color/depth extents or path counts shall be rejected.
```

```{spec} Native depth attachment and GPU transfers
:id: SPEC_DEPTH_ATTACHMENT
:status: implemented
:implements: REQ_DEPTH_SHARED, REQ_DEPTH_CONTROLS, REQ_DEPTH_PATHS

`DepthAttachment` owns `D32_SFLOAT` images left in
`DEPTH_STENCIL_ATTACHMENT_OPTIMAL`. Render passes load and store them using
the configured `DepthState`, comparing with `LESS_OR_EQUAL` when testing.
Writing without testing uses `ALWAYS`. Built-in draw components use the
render pass's pipeline depth state. Color producers report their final layout
so later writers can preserve the image while transitioning it.

Compute splatting reads GPU buffer snapshots; raster splatting uses native
read-only depth testing. Gaussian surface updates use GPU-only transfers.
The graph orders snapshots after the preceding color producer and attachment
updates after the depth calculation. No per-frame CPU readback is used.
```

```{spec} Gaussian depth selection
:id: SPEC_DEPTH_METHODS
:status: implemented
:implements: REQ_DEPTH_METHODS

`AccumulatedOpacity` selects the first depth where cumulative opacity reaches
`depthOpacityThreshold`. `NearestOpaqueSplat` selects the nearest splat whose
individual per-pixel opacity reaches that threshold. The default threshold
is 0.5; valid thresholds are in `(0, 1]`. `BackendDefault` selects accumulated
opacity on compute and nearest opaque splat on raster. `outputDepth` exports
a float buffer with positive infinity where no Gaussian surface was selected.
The shared attachment instead retains its previous value where no surface was
selected, and clears to 1 by default.

Raster accumulated-opacity depth is computed by a depth-only compute pipeline;
nearest-splat depth uses fragment atomics where supported, with a compute
fallback otherwise. Each method is an approximation to a translucent surface,
so different rendering orders need not give identical translucent blending.
```

```{test} Raster and Gaussian rendering order
:id: TEST_DEPTH_ORDER
:status: verified
:verifies: SPEC_DEPTH_ATTACHMENT, REQ_DEPTH_SHARED
:gtest: DepthComposition.gaussianAndRasterPassesComposeInEitherOrder

Checks nearer and farther opaque geometry in both rendering orders, with both
Gaussian backends and both depth methods.
```

```{test} Independent test and write controls
:id: TEST_DEPTH_CONTROLS
:status: verified
:verifies: SPEC_DEPTH_ATTACHMENT, REQ_DEPTH_CONTROLS
:gtest: DepthComposition.gaussianDepthControlsAreIndependent, DepthComposition.nativeDepthControlsAreIndependent

Disables testing and writing separately and verifies the resulting occlusion.
```

```{test} Clearing, reuse and separate paths
:id: TEST_DEPTH_PATHS
:status: verified
:verifies: SPEC_DEPTH_ATTACHMENT, REQ_DEPTH_PATHS
:gtest: DepthComposition.depthClearsBetweenFramesAndPathsStaySeparate, DepthComposition.preservesDepthWhenClearingIsDisabled, DepthComposition.rejectsMismatchedExtents, DepthComposition.rejectsMismatchedPathCounts

Reuses compiled graphs with moved Gaussians and two paths, verifies clearing
and independent color results, verifies explicit depth preservation, and rejects
mismatched attachment extents and path counts.
```

```{test} Both methods on both backends
:id: TEST_DEPTH_METHODS
:status: verified
:verifies: SPEC_DEPTH_METHODS, REQ_DEPTH_METHODS
:gtest: GaussianSplatDepthTest.depthMethodsAreSelectableOnBothBackends

An overlapping translucent stack has an accumulated-opacity surface and no
individually opaque-enough splat, with the same selected method on either backend.
```
