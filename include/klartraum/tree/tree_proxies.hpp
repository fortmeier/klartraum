// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_TREE_TREE_PROXIES_HPP
#define KLARTRAUM_TREE_TREE_PROXIES_HPP

#include <memory>
#include <vector>

#include "klartraum/storage_draw_component.hpp"
#include "klartraum/tree/tree_render_inputs.hpp"
#include "klartraum/tree/tree_tiles.hpp"
#include "klartraum/vulkan_gaussian_splatting_types.hpp"

namespace klartraum {

/// Push constants of `tree_place_cull.comp`.
struct TreeProxyCullPush {
    uint32_t capacity = 0;
    float minPixels = 0.0f;
};

struct TreeProxyConfig {
    uint32_t capacity = 65536; ///< visible trees per frame, at most
    float minPixels = 4.0f;    ///< trees smaller on screen are left out
};

/**
 * @brief The trees of a TreePlacementTiles pool that are visible this frame, drawn as simple proxies until their
 *        levels of detail exist.
 *
 * `tree_place_cull.comp` culls the pool's tiles and trees (view, size in
 * pixels) into a list of visible TreeSite; `tree_place_args.comp` writes the
 * indirect draw of the classical proxies (getDrawComponent(): a trunk and a
 * crown of two cones, 72 vertices each) and the dispatch of their Gaussians
 * (TreeProxyGaussians). Add getElement() as a compute dependency.
 */
class TreeProxies {
public:
    TreeProxies(VulkanContext& vulkanContext, TreePlacementTiles& tiles, std::shared_ptr<CameraUboType> camera,
                std::shared_ptr<HostValues<TreeRenderSettings>> settings, VkExtent2D extent,
                TreeRenderInputs inputs = {}, TreeProxyConfig config = {});

    static constexpr uint32_t kVertices = 72;

    ComputeGraphElementPtr getElement() const { return args; }
    std::shared_ptr<StorageDrawComponent> getDrawComponent() const { return draw; }
    BufferRef getVisible() const { return {cull, 6}; }
    BufferRef getCounter() const { return {cull, 7}; }
    std::shared_ptr<DispatchIndirectCommandBufferElement> getDispatchArgs() const { return dispatchArgs; }
    const TreeProxyConfig& getConfig() const { return config; }

    /// After a submission of `pathId`: [0] trees visible, [1] trees left out for lack of room.
    std::vector<uint32_t> readCounter(uint32_t pathId = 0);
    /// After a submission of `pathId`: the visible trees.
    std::vector<TreeSite> readVisible(uint32_t pathId = 0);

private:
    TreeProxyConfig config;
    std::shared_ptr<GeneralComputation<TreeProxyCullPush>> cull;
    std::shared_ptr<GeneralComputation<glm::uvec2>> args;
    std::shared_ptr<BufferElement<VulkanBuffer<TreeSite>>> visible;
    std::shared_ptr<BufferElement<VulkanBuffer<uint32_t>>> counter;
    std::shared_ptr<DispatchIndirectCommandBufferElement> dispatchArgs;
    std::shared_ptr<StorageDrawComponent> draw;
};

/**
 * @brief The visible proxy trees of TreeProxies as Gaussians: four per tree (trunk and crown), into `poolTrees` *
 *        4 slots from `poolOffset` of `gaussians`, cleared every frame.
 */
class TreeProxyGaussians {
public:
    TreeProxyGaussians(VulkanContext& vulkanContext, TreeProxies& proxies, const GaussianSoABuffers& gaussians,
                       uint32_t poolOffset, uint32_t poolTrees,
                       std::shared_ptr<HostValues<TreeRenderSettings>> settings, std::shared_ptr<CameraUboType> camera,
                       BufferRef light = {});

    const GaussianSoABuffers& getGaussians() const { return gaussians; }
    ComputeGraphElementPtr getElement() const { return write; }

private:
    std::shared_ptr<GeneralComputation<glm::uvec2>> clear;
    std::shared_ptr<GeneralComputation<glm::uvec2>> write;
    GaussianSoABuffers gaussians;
};

} // namespace klartraum

#endif // KLARTRAUM_TREE_TREE_PROXIES_HPP
