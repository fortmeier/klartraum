// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/tree/tree_terrain_gaussians.hpp"

#include "klartraum/storage_draw_component.hpp"

namespace klartraum {

TreeTerrainGaussians::TreeTerrainGaussians(VulkanContext& vulkanContext, TreeTerrain& terrain,
                                           const GaussianSoABuffers& source, uint32_t poolOffset, uint32_t poolSlots,
                                           std::shared_ptr<HostValues<TreeRenderSettings>> settings,
                                           std::shared_ptr<CameraUboType> camera, BufferRef light) {
    stats = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanContext, 4);
    stats->setName("TreeTerrainGaussianStats");
    stats->setRecordToZero(true);

    clear = std::make_shared<GeneralComputation<glm::uvec2>>(vulkanContext, "shaders/tree/tree_lod_clear.comp.spv");
    clear->setName("TreeTerrainGaussianClear");
    source.colAlpha.connectTo(*clear, 0);
    clear->setGroupCountX((poolSlots + 255) / 256);
    clear->setPushConstants({glm::uvec2(poolOffset, poolSlots)});

    write =
        std::make_shared<GeneralComputation<glm::uvec2>>(vulkanContext, "shaders/tree/tree_terrain_gaussians.comp.spv");
    write->setName("TreeTerrainGaussians");
    source.pos.connectTo(*write, 0);
    source.rot.connectTo(*write, 1);
    source.scale.connectTo(*write, 2);
    write->setInput(clear, 3, 0);
    terrain.getNodes().connectTo(*write, 4);
    terrain.getConfigBuffer().connectTo(*write, 5);
    terrain.getVertices().connectTo(*write, 6);
    terrain.getCovers().connectTo(*write, 7);
    write->setInput(settings, 8);
    write->setInput(stats, 9);
    const BufferRef lightRef = bufferOrPlaceholder(vulkanContext, light);
    write->setInput(lightRef.element, 10, lightRef.slot);
    write->setInput(camera, 11);
    terrain.getExtras().connectTo(*write, 12);
    write->setDynamicGroupDispatchParams(terrain.getDispatchArgs());
    write->setPushConstants({glm::uvec2(poolOffset, poolSlots)});

    gaussians = source;
    gaussians.pos = BufferRef{write, 0};
    gaussians.rot = BufferRef{write, 1};
    gaussians.scale = BufferRef{write, 2};
    gaussians.colAlpha = BufferRef{write, 3};
}

uint32_t TreeTerrainGaussians::readOverflow(uint32_t pathId) {
    std::vector<uint32_t> values(4, 0u);
    stats->getBuffer(pathId).memcopyTo(values);
    return values[0];
}

} // namespace klartraum
