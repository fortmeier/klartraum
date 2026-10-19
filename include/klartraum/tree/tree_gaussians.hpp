// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_TREE_TREE_GAUSSIANS_HPP
#define KLARTRAUM_TREE_TREE_GAUSSIANS_HPP

#include <memory>
#include <vector>

#include "klartraum/computegraph/generalcomputation.hpp"
#include "klartraum/computegraph/hostvalues.hpp"
#include "klartraum/draw_component.hpp"
#include "klartraum/storage_draw_component.hpp"
#include "klartraum/tree/tree_generator.hpp"
#include "klartraum/vulkan_gaussian_splatting_types.hpp"

namespace klartraum {

/**
 * @brief Turns the trees of a TreeGenerator into 3D Gaussians every frame, for the Gaussian splatting backends.
 *
 * `tree_gaussians.comp` writes kStemGaussians flat Gaussians on the tube of
 * every stem segment and kLeafGaussians along the lobes of every leaf, into
 * buffers sized for the generator's capacities; unused slots get opacity 0,
 * as do the Gaussians of stems thinner than TreeRenderSettings::viewport.w
 * pixels for `camera`, which the classical renderer leaves out as well.
 * The colours carry the lighting of TreeRenderSettings, with the shadows of
 * the `light` volume (TreeLightVolume) if given, as degree 0 spherical
 * harmonics; the higher bands are zero. `staticGaussians` (e.g. captured
 * objects) come first in the buffers, their colours tinted every frame by
 * TreeRenderSettings::capturedTint; `surfels`
 * (samples of static surfaces, see sampleMeshSurfels()) follow and are lit
 * every frame like the trees, so the trees' shadows fall on them. One backend
 * then sorts everything together without merging every frame. Call
 * initializeBuffers() once after the graph is compiled.
 */
class TreeGaussians {
public:
    TreeGaussians(VulkanContext& vulkanContext, TreeGenerator& generator, std::shared_ptr<CameraUboType> camera,
                  std::shared_ptr<HostValues<TreeRenderSettings>> settings,
                  std::vector<Gaussian3D> staticGaussians = {}, std::vector<TreeSceneVertex> surfels = {},
                  BufferRef light = {});

    static constexpr uint32_t kStemGaussians = 36;
    static constexpr uint32_t kLeafGaussians = 8;

    const GaussianSoABuffers& getGaussians() const { return gaussians; }
    ComputeGraphElementPtr getElement() const { return element; }

    /// Uploads the static Gaussians into every path and zeroes the trees' SH bands 1 to 3; call after compiling.
    void initializeBuffers(uint32_t numberPaths);

    /// Index of the first tree Gaussian.
    uint32_t getTreeOffset() const { return uint32_t(staticGaussians.size()) + surfelCount; }

private:
    std::shared_ptr<GeneralComputation<glm::uvec4>> element;
    GaussianSoABuffers gaussians;
    std::vector<Gaussian3D> staticGaussians;
    uint32_t surfelCount = 0;
};

/**
 * @brief Tints the colours of captured Gaussians every frame by TreeRenderSettings::capturedTint.
 *
 * Returns `gaussians` with the colours replaced by per-path buffers that
 * `tint_gaussians.comp` writes; `element` is set to the tinting element.
 */
GaussianSoABuffers tintCapturedGaussians(VulkanContext& vulkanContext, const GaussianSoABuffers& gaussians,
                                         std::shared_ptr<HostValues<TreeRenderSettings>> settings,
                                         ComputeGraphElementPtr& element);

} // namespace klartraum

#endif // KLARTRAUM_TREE_TREE_GAUSSIANS_HPP
