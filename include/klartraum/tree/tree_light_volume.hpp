// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_TREE_TREE_LIGHT_VOLUME_HPP
#define KLARTRAUM_TREE_TREE_LIGHT_VOLUME_HPP

#include <memory>

#include "klartraum/computegraph/generalcomputation.hpp"
#include "klartraum/computegraph/hostvalues.hpp"
#include "klartraum/tree/tree_generator.hpp"
#include "klartraum/rendering/vulkan_gaussian_splatting_types.hpp"

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
 * Static occluders (a pot, a captured object) add their optical depth from a
 * grid computed once; they cast shadows on the trees and each other, but not
 * on themselves. Captured objects, whose colours already contain their own
 * occlusion, read the light through the trees only. The grid's size is fixed
 * when the graph is compiled.
 */
class TreeLightVolume {
public:
    /// `staticDepth`: the optical depth per voxel of static occluders (treeStaticOcclusion()), or empty for none.
    TreeLightVolume(VulkanContext& vulkanContext, TreeGenerator& generator,
                    std::shared_ptr<HostValues<TreeRenderSettings>> settings, std::vector<float> staticDepth = {});

    BufferRef getLight() const { return {march, 2}; }
    ComputeGraphElementPtr getElement() const { return march; }

    /// Reads back the light of path `pathId` after its submission finished: per voxel the sun transmittance and sky
    /// visibility through all occluders (xy) and through the trees only (zw).
    std::vector<glm::vec4> readLight(uint32_t pathId = 0);

private:
    std::shared_ptr<GeneralComputation<glm::uvec2>> density;
    std::shared_ptr<GeneralComputation<>> march;
    std::shared_ptr<BufferElement<VulkanBuffer<glm::vec4>>> light;
};

/**
 * @brief Optical depth per voxel of the light grid of `settings` of opaque static surfaces and Gaussians.
 *
 * Surfels (TreeSceneVertex with their spacing in pos.w, see sampleMeshSurfels())
 * add their area, Gaussians their opacity times the area of their two larger
 * axes; a voxel whose face is covered gets `opaqueDepth`, so light that
 * crosses a solid surface is nearly blocked.
 */
std::vector<float> treeStaticOcclusion(const TreeRenderSettings& settings, const std::vector<TreeSceneVertex>& surfels,
                                       const std::vector<Gaussian3D>& gaussians, float opaqueDepth = 3.0f);

} // namespace klartraum

#endif // KLARTRAUM_TREE_TREE_LIGHT_VOLUME_HPP
