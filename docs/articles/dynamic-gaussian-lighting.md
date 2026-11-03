# Dynamic lighting of Gaussian scenes

Klartraum applies dynamic point lights to a Gaussian scene with
{cpp:class}`klartraum::GaussianLighting`, a compute-graph operation that runs
before splatting. A CPU bake supplies approximate surface normals and
directional visibility. Each frame, a compute shader combines these materials
with the current lights and camera position, then writes a color buffer for
either Gaussian rendering backend.

The input colors come from a captured scene. They already contain illumination;
Klartraum uses them as an approximation of albedo rather than recovering the
surface reflectance of the original object. This makes the operation useful
for art-directed relighting, with limitations around captured shadows and
highlights.

## Where lighting runs

The graph has this data flow:

```text
Gaussian positions and captured colors ──┐
Baked SplatMaterial buffer ───────────────┼─ GaussianLighting ─ lit colors ─ splatting
HostValues<LightingFrame> ────────────────┘
Other Gaussian attributes ─────────────────────────────────────────────── splatting
```

`createGaussianLighting()` uploads one {cpp:struct}`klartraum::SplatMaterial`
per Gaussian into a device-local buffer shared by the graph's execution paths.
It allocates a separate output color buffer for each path. The returned
`output` uses this color buffer and forwards the input positions, rotations,
scales and higher spherical-harmonic color coefficients. Connecting `output`
to `createGaussianSplatting()` establishes the graph dependency that runs
lighting before rendering.

The graph records the dispatches once. Updating
{cpp:class}`klartraum::HostValues` changes the lighting parameters without
recording the command buffers again. `set()` stores values on the CPU;
immediately before a path is submitted, its update copies those values into
that path's GPU buffer. Other paths can retain their in-flight values.

## Baking normals and visibility

{cpp:func}`klartraum::bakeSplatMaterials` takes the CPU-side Gaussian scene.
Its default grid has 24 cells per side. The grid uses cubic cells sized from
the largest span between the first and ninety-ninth percentiles of the
Gaussian positions, with padding. Outlying positions are clamped into edge
cells, so a few distant splats do not determine the grid resolution.

Each Gaussian adds opacity to its cell, weighted by its alpha and largest
cross-sectional area. Accumulated opacity is converted to a value between zero
and one. For each occupied cell, the bake samples transmittance along 64
directions distributed over a sphere, excluding the starting cell. It
projects these samples onto nine real spherical harmonics: degrees zero
through two. Gaussians in the same cell share these visibility coefficients.

Visibility in a direction is the SH reconstruction clamped to `[0, 1]`.
The constant coefficient also supplies the average visibility used for ambient
light. These are material visibility coefficients, separate from the
Gaussian's captured color SH coefficients.

A Gaussian's shortest local scale axis provides its approximate surface
normal. The bake rotates that axis by the Gaussian's quaternion and chooses
the sign that faces greater visibility. Roughness, metalness and specular scale
are supplied to the bake; their defaults are `0.8`, `0` and `1`. Applications
can also provide their own materials, preserving one material per Gaussian in
the input order.

## Shading a Gaussian

The lighting shader dispatches one invocation per Gaussian, in workgroups of
256. It decodes the captured degree-zero color coefficient using
`max(0.282094792 * coefficient + 0.5, 0)` as albedo.

Ambient light contributes albedo multiplied by `LightingFrame::ambient.rgb`
and average visibility. `ambient.w` is unused. Up to eight point lights then
contribute diffuse and specular illumination:

- Diffuse uses a Lambert term, scaled by `1 - metalness`.
- Specular uses a GGX distribution, a Schlick-style geometry term and Schlick
  Fresnel. Roughness is clamped to a minimum of `0.05`. The normal-incidence
  reflectance interpolates from `0.04` to albedo with metalness.
- Each direct-light contribution is multiplied by the positive normal/light
  cosine, directional visibility, light color, intensity and attenuation.

The camera position controls the specular viewing direction. For splats with
alpha below `0.8`, the shader flips the normal towards the camera when needed,
allowing thin translucent splats to receive light on either face.

### Distance, range and visibility

A {cpp:struct}`klartraum::SplatLight` packs its parameters into three vectors:

| Field | Meaning |
|---|---|
| `positionRange.xyz` | Light position in the same coordinates as the Gaussians and material normals |
| `positionRange.w` | Cutoff range; zero or negative means unbounded |
| `colourIntensity.rgb`, `.w` | Light color and intensity |
| `shape.x` | Reference distance for attenuation |
| `shape.y` | Radius that clamps the denominator of distance attenuation |
| `shape.z` | Values above `0.5` bypass baked visibility for direct lighting |
| `shape.w` | Glow radius; zero disables glow |

Distance attenuation is `reference² / max(distance², radius²)`, with a small
distance floor in the shader. At the reference distance the distance factor
is one when the radius does not clamp it. The final surface brightness still
depends on the BRDF, orientation and visibility. For a positive range, the
factor is also multiplied by `(1 - (distance / range)^4)^2` inside the range;
at and beyond the range the light contributes nothing.

Directional visibility comes from the material's SH evaluated towards the
light. It contains no light-distance information. The unshadowed setting
bypasses that visibility but retains the normal/light cosine and BRDF; it
does not illuminate a back-facing opaque splat by itself.

### Glow and the relight blend

Inside a positive glow radius, a light adds
`color * intensity * (1 - distance / glowRadius)^2`. This addition is
independent of albedo, normal direction and visibility. The light's range
cutoff still applies. It can suggest emission inside a lantern, but it does
not perform bloom or illuminate other splats through bounced light.

`LightingFrame::control.x` blends between captured and shaded degree-zero
colors and is clamped to `[0, 1]`. At zero, the shader copies the original
color/alpha vector exactly. At one, it encodes the shaded result as the
degree-zero coefficient. Alpha remains unchanged throughout. Higher color SH
bands pass through unchanged, so their view-dependent captured appearance
can remain visible when evaluated by the rendering backend.

## Animating a light in a recorded graph

This example assumes an initialized Vulkan context, a nonempty scene, an
initialized camera and an offscreen target with one execution path. Positions
in the example are in scene coordinates; adjust the orbit and intensity to
the scale of the model. The caller can read back the target after rendering.

```cpp
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/hostvalues.hpp"
#include "klartraum/offscreen_target.hpp"
#include "klartraum/rendering/gaussian_data_standard.hpp"
#include "klartraum/rendering/gaussian_lighting.hpp"
#include "klartraum/rendering/gaussian_splatting_factory.hpp"

void animateLight(klartraum::VulkanContext& vc,
                  const std::vector<klartraum::Gaussian3D>& scene,
                  std::shared_ptr<klartraum::CameraUboType> camera,
                  std::shared_ptr<klartraum::OffscreenTarget> target) {
    using namespace klartraum;
    auto data = std::make_shared<GaussianDataStandard>(vc, scene);
    auto frame = std::make_shared<HostValues<LightingFrame>>(vc);
    const auto materials = bakeSplatMaterials(scene);
    auto lighting = createGaussianLighting(vc, data->buffers(), materials, BufferRef{frame});
    auto splatting = createGaussianSplatting(vc, GsplatBackend::Compute,
                                           target, camera, lighting.output);
    ComputeGraph graph(vc, 1);
    graph.compileFrom(splatting);

    LightingFrame values;
    values.control.x = 1.0f;
    values.ambient = glm::vec4(0.15f, 0.15f, 0.15f, 0.0f);
    values.counts.x = 1;
    values.lights[0].colourIntensity = {1.0f, 0.7f, 0.4f, 2.0f};
    values.lights[0].shape = {1.0f, 0.05f, 0.0f, 0.0f};

    for (uint32_t i = 0; i < 120; ++i) {
        const float angle = 6.2831853f * float(i) / 120.0f;
        values.lights[0].positionRange = {std::cos(angle), 0.5f,
                                         std::sin(angle), 0.0f};
        values.camera = glm::vec4(glm::vec3(camera->ubo.cameraWorldPos), 0.0f);
        camera->update(0);
        frame->set(0, values);
        graph.submitAndWait(vc.getGraphicsQueue(), 0);
    }
}
```

`submitAndWait()` keeps this headless example synchronous. A frontend-driven
application sets the same frame values before its next frame submission and
uses the frontend's normal path scheduling. Update the lighting camera value
when the rendering camera moves. Material buffers stay fixed while lights,
ambient color and the relight blend change.

## Limits of the model

The material bake represents static geometry. Moving a light changes which
direction of baked visibility is sampled, but moving an occluder does not
update that visibility. The coarse grid and degree-two SH smooth away small
openings and sharp shadow boundaries. Occluders beyond a nearby point light
can still reduce its visibility because the bake samples directions through
the scene rather than stopping at the light.

Normals and visibility must match the positions used by the lighting pass.
Transforming or morphing positions alone does not transform material normals
or rebake occlusion. Keep all inputs in a consistent coordinate system and
provide appropriate materials for the geometry being shaded.

Captured illumination remains in the approximate albedo and in the forwarded
higher color SH bands. There is no inverse-rendering step, dynamic shadow map,
indirect-light transport, area-light integration or bloom in this operation.
The tests in `tests/test_gaussian_lighting.cpp` check the visibility basis,
open and enclosed scenes, normal orientation, captured-color preservation,
ambient scaling, Lambert response, unshadowed lights, glow and a rendered
lantern comparison. They exercise the implemented approximation rather than
establishing physical accuracy for arbitrary captured scenes.
