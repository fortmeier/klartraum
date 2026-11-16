// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_TREE_TREE_LIGHT_VOLUME_HPP
#define KLARTRAUM_TREE_TREE_LIGHT_VOLUME_HPP

#include <memory>
#include <vector>

#include "klartraum/computegraph/generalcomputation.hpp"
#include "klartraum/computegraph/hostvalues.hpp"
#include "klartraum/tree/tree_generator.hpp"
#include "klartraum/rendering/vulkan_gaussian_splatting_types.hpp"

namespace klartraum {

/// Sets the light volume of `settings` to cover [lower, upper] with cubic voxels of `voxel` units, and enables it.
void configureTreeLightGrid(TreeRenderSettings& settings, glm::vec3 lower, glm::vec3 upper, float voxel,
                            float opticalDepthPerArea = 0.5f);

/// Occluders that move, such as a falling object: samples of their surfaces in each object's own frame, placed
/// every frame by the object's transform (TreeLightVolume::setMovingTransform()).
struct TreeMovingOccluders {
    std::vector<glm::vec4> surfels; ///< xyz position in the object's frame, w the area it stands for
    std::vector<uint32_t> objects;  ///< the object of each surfel, below objectCount
    uint32_t objectCount = 0;
};

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
 * on themselves. Moving occluders (TreeMovingOccluders) are added every frame
 * by `tree_light_movers.comp` at their current transforms and count like the
 * static ones. The grid's size is fixed when the graph is compiled.
 */
class TreeLightVolume {
public:
    /// `staticDepth`: the optical depth per voxel of static occluders (treeStaticOcclusion()), or empty for none;
    /// `moving`: the occluders that move, each at the identity until setMovingTransform() places it.
    TreeLightVolume(VulkanContext& vulkanContext, TreeGenerator& generator,
                    std::shared_ptr<HostValues<TreeRenderSettings>> settings, std::vector<float> staticDepth = {},
                    TreeMovingOccluders moving = {});
    /// Places moving occluder `object` for the next submissions; a transform of scale 0 removes it.
    void setMovingTransform(uint32_t object, const glm::mat4& worldFromObject);
    /**
     * @brief Adds an occluder read from `vertices` every frame (four floats each, xyz in the object's frame, e.g. a
     * MeshTraining's positions), each sample standing for `area` (in the object's frame), placed by moving occluder
     * `object`'s transform; read after `after`, e.g. the element writing the vertices. Before compiling the graph.
     */
    void addLiveOccluder(const BufferRef& vertices, uint32_t count, float area, uint32_t object,
                         const ComputeGraphElementPtr& after = nullptr);

    BufferRef getLight() const { return {march, 2}; }
    ComputeGraphElementPtr getElement() const { return march; }

    /// Reads back the light of path `pathId` after its submission finished: per voxel the sun transmittance and sky
    /// visibility through all occluders (xy) and through the trees only (zw).
    std::vector<glm::vec4> readLight(uint32_t pathId = 0);

private:
    /// Push constants of `tree_light_movers.comp`.
    struct TreeMoversPush {
        uint32_t count;  ///< samples
        uint32_t object; ///< the samples' object, or kPerSample to read it per sample
        float area;      ///< the area of every sample, or 0 to read it per sample
        uint32_t unused;
    };
    static constexpr uint32_t kPerSample = 0xffffffffu;

    std::shared_ptr<GeneralComputation<glm::uvec2>> density;
    // The transforms of the moving occluders (one at least, unused without them).
    std::shared_ptr<HostValues<glm::mat4>> movingTransforms;
    // The moving occluders' grid as the last pass adding to it leaves it, which the march reads.
    BufferRef movingGrid;
    uint32_t movingObjectCount = 0;
    // For adding live occluders after construction.
    VulkanContext* vulkanContext = nullptr;
    std::shared_ptr<HostValues<TreeRenderSettings>> settings;
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

/// The surfels and Gaussians of an occluder as moving occluder samples (TreeMovingOccluders::surfels), with the
/// areas treeStaticOcclusion() gives them.
std::vector<glm::vec4> treeOccluderSamples(const std::vector<TreeSceneVertex>& surfels,
                                           const std::vector<Gaussian3D>& gaussians);

} // namespace klartraum

#endif // KLARTRAUM_TREE_TREE_LIGHT_VOLUME_HPP
