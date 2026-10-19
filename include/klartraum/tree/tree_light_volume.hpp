// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_TREE_TREE_LIGHT_VOLUME_HPP
#define KLARTRAUM_TREE_TREE_LIGHT_VOLUME_HPP

#include <memory>

#include "klartraum/computegraph/generalcomputation.hpp"
#include "klartraum/computegraph/hostvalues.hpp"
#include "klartraum/tree/tree_generator.hpp"

namespace klartraum {

/// Sets the light volume of `settings` to cover [lower, upper] with cubic voxels of `voxel` units, and enables it.
void configureTreeLightGrid(TreeRenderSettings& settings, glm::vec3 lower, glm::vec3 upper, float voxel,
                            float opticalDepthPerArea = 0.5f);

/**
 * @brief Soft shadows and sky occlusion of the generated trees, for both the classical and the Gaussian renderer.
 *
 * Every frame `tree_light_density.comp` adds the area of every leaf and stem
 * segment into a voxel grid (TreeRenderSettings::lightGridMin/Size), and
 * `tree_light_march.comp` marches from every voxel towards the sun and five
 * sky directions through it. The result (getLight(): per voxel the sun
 * transmittance and the sky visibility) is what `tree_light.glsl` reads.
 * The grid's size is fixed when the graph is compiled.
 */
class TreeLightVolume {
public:
    TreeLightVolume(VulkanContext& vulkanContext, TreeGenerator& generator,
                    std::shared_ptr<HostValues<TreeRenderSettings>> settings);

    BufferRef getLight() const { return {march, 2}; }
    ComputeGraphElementPtr getElement() const { return march; }

    /// Reads back the light of path `pathId` (two floats per voxel) after its submission finished.
    std::vector<glm::vec2> readLight(uint32_t pathId = 0);

private:
    std::shared_ptr<GeneralComputation<glm::uvec2>> density;
    std::shared_ptr<GeneralComputation<>> march;
    std::shared_ptr<BufferElement<VulkanBuffer<glm::vec2>>> light;
};

} // namespace klartraum

#endif // KLARTRAUM_TREE_TREE_LIGHT_VOLUME_HPP
