// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_TREE_TREE_IMPOSTORS_HPP
#define KLARTRAUM_TREE_TREE_IMPOSTORS_HPP

#include <memory>
#include <vector>

#include "klartraum/storage_draw_component.hpp"
#include "klartraum/tree/tree_proxies.hpp"
#include "klartraum/tree/tree_render_inputs.hpp"

namespace klartraum {

struct TreeImpostorConfig {
    uint32_t views = 8;   ///< per side of the hemi-octahedral grid of view directions
    uint32_t pixels = 64; ///< per side of a view, baked at twice the resolution and filtered down
    std::vector<float> seasons{0.2f, 1.2f, 2.0f, 3.0f}; ///< the seasons S of the samples, ascending, at most 4
};

/**
 * @brief Octahedral impostors of the tree types: each type, generated once per season sample, seen from the views
 *        of a hemi-octahedral grid (treeHemiOctDecode()), its albedo, coverage and normals in atlases.
 *
 * Baking renders all views of a type at once: copies of the tree, each
 * turned so that an orthographic camera looking along -z sees it from its
 * view, with up on the screen as treeImpostorUp() (so the normals come out in
 * the view's frame: x right, y up, z towards the viewer); the stem and leaf
 * renderers write albedo and normals instead of colours
 * (TreeRenderSettings::occlusion.y). Per texel: albedo as RGBA8 (rgb the
 * square root of the albedo, a the coverage), normal as RGB8 (0.5 n + 0.5).
 */
class TreeImpostorAtlas {
public:
    TreeImpostorAtlas(VulkanContext& vulkanContext, const std::vector<TreeTypeParams>& types,
                      TreeImpostorConfig config = {});

    const TreeImpostorParams& getParams() const { return params; }
    /// Texels (rows from the top) of type `type` and season sample `season`, views x pixels on a side.
    const std::vector<uint32_t>& getAlbedo() const { return albedo; }
    const std::vector<uint32_t>& getNormals() const { return normals; }
    uint32_t side() const { return params.grid.x * params.grid.y; }
    size_t layer(uint32_t type, uint32_t season) const {
        return (size_t(type) * params.grid.z + season) * side() * side();
    }

    std::shared_ptr<BufferElementSinglePath<VulkanBuffer<uint32_t>>> getAlbedoBuffer() const { return albedoBuffer; }
    std::shared_ptr<BufferElementSinglePath<VulkanBuffer<uint32_t>>> getNormalBuffer() const { return normalBuffer; }
    std::shared_ptr<BufferElementSinglePath<VulkanBuffer<TreeImpostorParams>>> getParamsBuffer() const {
        return paramsBuffer;
    }

private:
    TreeImpostorParams params{};
    std::vector<uint32_t> albedo;
    std::vector<uint32_t> normals;
    std::shared_ptr<BufferElementSinglePath<VulkanBuffer<uint32_t>>> albedoBuffer;
    std::shared_ptr<BufferElementSinglePath<VulkanBuffer<uint32_t>>> normalBuffer;
    std::shared_ptr<BufferElementSinglePath<VulkanBuffer<TreeImpostorParams>>> paramsBuffer;
};

/**
 * @brief Draws trees (TreeSite, e.g. the visible ones of TreeProxies) as impostors: a card per tree facing the
 *        camera, its texels blended from the four nearest views and the two nearest season samples of a
 *        TreeImpostorAtlas, relit with the baked normals; coverage and the trees' fade (kind.w) by screen-door
 *        dithering.
 *
 * `drawArgs` holds a VkDrawIndirectCommand of 6 vertices per tree at `offset`
 * (TreeProxies::getDrawArgs() has it second).
 */
class TreeImpostorRenderer {
public:
    TreeImpostorRenderer(VulkanContext& vulkanContext, BufferRef sites, BufferRef drawArgs, VkDeviceSize offset,
                         const TreeImpostorAtlas& atlas, std::shared_ptr<HostValues<TreeRenderSettings>> settings,
                         VkExtent2D extent, TreeRenderInputs inputs = {});

    std::shared_ptr<StorageDrawComponent> getDrawComponent() const { return draw; }

private:
    std::shared_ptr<StorageDrawComponent> draw;
};

} // namespace klartraum

#endif // KLARTRAUM_TREE_TREE_IMPOSTORS_HPP
