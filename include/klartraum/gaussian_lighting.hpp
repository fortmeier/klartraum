// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_GAUSSIAN_LIGHTING_HPP
#define KLARTRAUM_GAUSSIAN_LIGHTING_HPP

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include <glm/glm.hpp>

#include "klartraum/computegraph/generalcomputation.hpp"
#include "klartraum/vulkan_gaussian_splatting_types.hpp"

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

/// A point light. Its intensity falls off with the squared distance beyond
/// `reference` (at the reference distance a surface facing it receives
/// colour x intensity); within `range` > 0 it fades smoothly to zero. An
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
    glm::vec4 ambient{1.0f}; ///< rgb: ambient colour x strength
    glm::uvec4 counts{0u};   ///< x: number of lights
    glm::vec4 control{0.0f}; ///< x: relight blend (0: captured colours, 1: relit)
    std::array<SplatLight, kMaxLights> lights{};
};

static_assert(sizeof(SplatMaterial) == 64 && sizeof(SplatLight) == 48 && sizeof(LightingFrame) == 64 + 8 * 48);

/// The degree-2 real SH basis in the order of SplatMaterial.
std::array<float, 9> visibilityBasis(const glm::vec3& direction);
/// Visibility of a material towards `direction`, in [0, 1].
float evaluateVisibility(const SplatMaterial& material, const glm::vec3& direction);

/// Bakes the materials of a scene, once on the CPU. The scene is binned into a
/// grid of `gridSide`^3 cells (spanning the 1st to 99th percentile of the
/// positions, so stray splats do not stretch it); each cell gets an opacity
/// from the splats in it, and 64 rays per occupied cell give the transmittance
/// by direction, projected onto degree-2 SH. A splat's normal is its shortest
/// axis, turned to the side where the visibility is larger.
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
 * and the view-dependent SH bands pass through.
 */
class GaussianLighting : public GeneralComputation<glm::uvec4> {
public:
    GaussianLighting(VulkanContext& vulkanContext)
        : GeneralComputation<glm::uvec4>(vulkanContext, "shaders/gaussians/lighting.comp.spv") {}

    const char* getType() const override { return "GaussianLighting"; }
};

struct GaussianLightingResult {
    std::shared_ptr<GaussianLighting> element;
    GaussianSoABuffers output; ///< the input with the colour of input 4
};

/**
 * @brief Creates a GaussianLighting for @p input with @p materials (one per Gaussian) and the
 * LightingFrame buffer @p frame.
 */
GaussianLightingResult createGaussianLighting(VulkanContext& vulkanContext, const GaussianSoABuffers& input,
                                              const std::vector<SplatMaterial>& materials, const BufferRef& frame);

} // namespace klartraum

#endif // KLARTRAUM_GAUSSIAN_LIGHTING_HPP
