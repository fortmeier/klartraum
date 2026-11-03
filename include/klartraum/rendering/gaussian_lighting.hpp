// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_RENDERING_GAUSSIAN_LIGHTING_HPP
#define KLARTRAUM_RENDERING_GAUSSIAN_LIGHTING_HPP

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include <glm/glm.hpp>

#include "klartraum/computegraph/generalcomputation.hpp"
#include "klartraum/rendering/vulkan_gaussian_splatting_types.hpp"

namespace klartraum {

/// Surface properties of one Gaussian for relighting, independent of the light.
struct SplatMaterial {
    glm::vec4 normalRoughness{0.0f, 0.0f, 1.0f, 0.8f}; ///< xyz: unit normal (world), w: GGX roughness
    /// Degree-2 SH of the visibility (how open the surroundings are) by
    /// direction: coefficients 0-3, 4-7 and 8 (in parameters.x).
    glm::vec4 visibility0{3.5449077f, 0.0f, 0.0f, 0.0f}; ///< fully open: 1 / 0.282095
    glm::vec4 visibility1{0.0f};
    glm::vec4 parameters{0.0f, 0.0f, 1.0f, 0.0f}; ///< x: visibility coefficient 8, y: metalness, z: specular scale
};

/// A point light with inverse-square attenuation, clamped by its radius.
/// At the reference distance the distance attenuation is one unless the radius
/// clamps it; BRDF, orientation, visibility and range also affect shading.
/// Within `range` > 0 it fades smoothly to zero. An
/// unshadowed light ignores the baked visibility, e.g. a light inside an
/// object that shines out of openings the coarse visibility does not resolve.
/// Within the glow radius, splats emit colour x intensity, fading with the
/// squared distance to the edge of the radius, e.g. the glowing inside of a lamp.
struct SplatLight {
    glm::vec4 positionRange{0.0f};   ///< xyz: position, w: range (0: unbounded)
    glm::vec4 colourIntensity{0.0f}; ///< rgb: colour, w: intensity
    /// x: reference distance, y: radius (no closer than this), z: 1 = unshadowed, w: glow radius (0: no glow)
    glm::vec4 shape{1.0f, 0.0f, 0.0f, 0.0f};
};

/// Everything the lighting reads per frame (HostValues<LightingFrame>).
struct LightingFrame {
    static constexpr uint32_t kMaxLights = 8;

    glm::vec4 camera{0.0f};  ///< xyz: camera position, for the specular term
    glm::vec4 ambient{1.0f}; ///< rgb: ambient colour x strength; w: unused
    glm::uvec4 counts{0u};   ///< x: number of lights, clamped to kMaxLights; yzw: unused
    glm::vec4 control{0.0f}; ///< x: relight blend, clamped to [0, 1]; yzw: unused
    std::array<SplatLight, kMaxLights> lights{}; ///< Point lights in the same coordinates as the Gaussian positions.
};

static_assert(sizeof(SplatMaterial) == 64 && sizeof(SplatLight) == 48 && sizeof(LightingFrame) == 64 + 8 * 48);

/// Evaluates the degree-2 real SH basis for a unit direction, in SplatMaterial order.
std::array<float, 9> visibilityBasis(const glm::vec3& direction);
/// Evaluates material visibility towards a nonzero direction, normalized internally, in [0, 1].
float evaluateVisibility(const SplatMaterial& material, const glm::vec3& direction);

/// Bakes the materials of a scene, once on the CPU. The scene is binned into a
/// grid of `gridSide`^3 cells (spanning the 1st to 99th percentile of the
/// positions, so stray splats do not stretch it); each cell gets an opacity
/// from the splats in it, and 64 rays per occupied cell give the transmittance
/// by direction, projected onto degree-2 SH. A splat's normal is its shortest
/// axis, turned to the side where the visibility is larger.
/// Materials retain the input order and describe static geometry; moving
/// occluders or transforming positions does not update normals or visibility.
/// An empty scene returns no materials; a nonempty scene requires gridSide >= 3.
std::vector<SplatMaterial> bakeSplatMaterials(const std::vector<Gaussian3D>& gaussians, float roughness = 0.8f,
                                              float metalness = 0.0f, float specularScale = 1.0f, int gridSide = 24);

/**
 * @brief Relights Gaussians every time the graph runs.
 *
 * Shades each Gaussian with its captured colour as albedo, its SplatMaterial
 * and the lights of a LightingFrame buffer (shaders/gaussians/lighting.comp):
 * ambient x albedo x average visibility, plus per light a Lambert and a GGX
 * term weighted by the visibility towards the light. The result replaces the
 * SH degree-0 colour, blended with the captured colour by the relight value;
 * at relight 0 the output is the input colour, unchanged. Positions, shapes
 * and the view-dependent SH bands pass through, as does alpha. Captured
 * illumination remains in the approximate albedo and the higher SH bands.
 * The shader supports up to eight point lights and optional local glow;
 * directional visibility describes the static geometry used for the bake.
 */
class GaussianLighting : public GeneralComputation<glm::uvec4> {
public:
    GaussianLighting(VulkanContext& vulkanContext)
        : GeneralComputation<glm::uvec4>(vulkanContext, "shaders/gaussians/lighting.comp.spv") {}

    const char* getType() const override { return "GaussianLighting"; }
};

/// Owns the lighting operation and references its rendered Gaussian attributes.
struct GaussianLightingResult {
    std::shared_ptr<GaussianLighting> element; ///< Graph operation producing the lit color buffer.
    GaussianSoABuffers output; ///< Input attributes with colAlpha replaced by the per-path lit color/alpha buffer.
};

/**
 * @brief Creates a GaussianLighting for @p input with @p materials (one per Gaussian) and the
 * LightingFrame buffer @p frame.
 *
 * Materials are uploaded once and shared by all paths; the color output is
 * per-path. Connect the returned output to a Gaussian renderer before graph
 * compilation. HostValues<LightingFrame> can update the frame parameters
 * before each submission without recompiling the graph. Positions, material
 * normals, light positions and camera positions must use the same coordinates.
 * @throws std::invalid_argument If the material count differs from input.count.
 */
GaussianLightingResult createGaussianLighting(VulkanContext& vulkanContext, const GaussianSoABuffers& input,
                                              const std::vector<SplatMaterial>& materials, const BufferRef& frame);

} // namespace klartraum

#endif // KLARTRAUM_RENDERING_GAUSSIAN_LIGHTING_HPP
