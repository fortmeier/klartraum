// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_TREE_BONSAI_RENDERER_HPP
#define KLARTRAUM_TREE_BONSAI_RENDERER_HPP

#include <memory>
#include <string>

#include "klartraum/computegraph/hostvalues.hpp"
#include "klartraum/computegraph/imageresample.hpp"
#include "klartraum/computegraph/imageviewsrc.hpp"
#include "klartraum/computegraph/renderpass.hpp"
#include "klartraum/gaussian_data_standard.hpp"
#include "klartraum/gaussian_splatting_factory.hpp"
#include "klartraum/offscreen_target.hpp"
#include "klartraum/tree/bonsai_scene.hpp"
#include "klartraum/tree/scene_mesh_renderer.hpp"
#include "klartraum/tree/tree_gaussians.hpp"
#include "klartraum/tree/tree_generator.hpp"
#include "klartraum/tree/tree_leaf_renderer.hpp"
#include "klartraum/tree/tree_light_volume.hpp"
#include "klartraum/tree/tree_stem_renderer.hpp"

namespace klartraum {

enum class BonsaiRenderMode {
    Classical, ///< rasterized trees and pot; the lantern splatted, the two composited by depth
    Gaussian,  ///< everything as Gaussians, sorted and splatted together
};

struct BonsaiRenderConfig {
    VkExtent2D extent{1280, 720};
    BonsaiRenderMode mode = BonsaiRenderMode::Classical;
    GsplatBackend backend = GsplatBackend::Raster;
    std::string lanternPath = "data/lantern.spz";
    float meshGaussianSpacing = 0.008f; ///< of the pot and soil Gaussians
    float lightVoxel = 0.035f;          ///< voxel size of the light volume (shadows and sky occlusion)
    uint32_t supersample = 2;           ///< the classical renderer draws at this times the resolution
    uint32_t numberPaths = 1;           ///< of the graph the renderer is compiled into
};

/**
 * @brief The compute graph that draws a BonsaiScene into a target, either way.
 *
 * Per frame only the camera, the TreeRenderSettings and the TreeFrame come
 * from the CPU; the trees are generated, their shadows computed in a
 * TreeLightVolume, culled, tessellated or turned into Gaussians, and drawn on
 * the GPU. Compile a graph from getOutput(), then call
 * initialize() once before the first submission.
 */
class BonsaiRenderer {
public:
    BonsaiRenderer(VulkanContext& vulkanContext, const BonsaiScene& scene, std::shared_ptr<ImageViewSrc> target,
                   const BonsaiRenderConfig& config);

    ComputeGraphElementPtr getOutput() const { return output; }
    void initialize(uint32_t numberPaths);

    std::shared_ptr<CameraUboType> getCamera() const { return camera; }
    HostValues<TreeRenderSettings>& getSettings() { return *settings; }
    TreeGenerator& getGenerator() { return *generator; }
    TreeStemRenderer* getStems() { return stems.get(); }
    TreeLeafRenderer* getLeaves() { return leaves.get(); }
    TreeLightVolume& getLightVolume() { return *lightVolume; }

    /// The lantern capture, cropped to the lantern and its moss and placed on the soil.
    static std::vector<Gaussian3D> loadLantern(const BonsaiScene& scene, const std::string& path);

private:
    BonsaiRenderConfig config;
    std::shared_ptr<CameraUboType> camera;
    std::shared_ptr<HostValues<TreeRenderSettings>> settings;
    std::shared_ptr<TreeGenerator> generator;
    std::unique_ptr<TreeLightVolume> lightVolume;

    // Classical
    std::shared_ptr<GaussianDataStandard> lantern;
    std::unique_ptr<TreeStemRenderer> stems;
    std::unique_ptr<TreeLeafRenderer> leaves;
    std::unique_ptr<SceneMeshRenderer> meshes;
    std::shared_ptr<RenderPass> pass;

    // Gaussian
    std::unique_ptr<TreeGaussians> gaussians;

    ComputeGraphElementPtr output;
};

} // namespace klartraum

#endif // KLARTRAUM_TREE_BONSAI_RENDERER_HPP
