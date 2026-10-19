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
 * The colours carry the lighting of TreeRenderSettings as degree 0 spherical
 * harmonics; the higher bands are zero. `staticGaussians` (e.g. a scene's
 * meshes and captured objects) come first in the buffers, so one backend
 * sorts everything together without merging every frame. Call
 * initializeBuffers() once after the graph is compiled.
 */
class TreeGaussians {
public:
    TreeGaussians(VulkanContext& vulkanContext, TreeGenerator& generator, std::shared_ptr<CameraUboType> camera,
                  std::shared_ptr<HostValues<TreeRenderSettings>> settings,
                  std::vector<Gaussian3D> staticGaussians = {});

    static constexpr uint32_t kStemGaussians = 16;
    static constexpr uint32_t kLeafGaussians = 8;

    const GaussianSoABuffers& getGaussians() const { return gaussians; }
    ComputeGraphElementPtr getElement() const { return element; }

    /// Uploads the static Gaussians into every path and zeroes the trees' SH bands 1 to 3; call after compiling.
    void initializeBuffers(uint32_t numberPaths);

    /// Index of the first tree Gaussian.
    uint32_t getTreeOffset() const { return uint32_t(staticGaussians.size()); }

private:
    std::shared_ptr<GeneralComputation<glm::uvec4>> element;
    GaussianSoABuffers gaussians;
    std::vector<Gaussian3D> staticGaussians;
};

} // namespace klartraum

#endif // KLARTRAUM_TREE_TREE_GAUSSIANS_HPP
