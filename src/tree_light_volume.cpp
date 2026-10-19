// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/tree/tree_light_volume.hpp"

#include <cmath>
#include <stdexcept>

namespace klartraum {

void configureTreeLightGrid(TreeRenderSettings& settings, glm::vec3 lower, glm::vec3 upper, float voxel,
                            float opticalDepthPerArea) {
    const glm::uvec3 size = glm::uvec3(glm::ceil((upper - lower) / voxel));
    settings.lightGridMin = glm::vec4(lower, voxel);
    settings.lightGridSize = glm::uvec4(size, 1u);
    settings.lightGridParams = glm::vec4(opticalDepthPerArea, 0.0f, 0.0f, 0.0f);
}

TreeLightVolume::TreeLightVolume(VulkanContext& vulkanContext, TreeGenerator& generator,
                                 std::shared_ptr<HostValues<TreeRenderSettings>> settings) {
    const TreeRenderSettings& values = settings->values().at(0);
    const glm::uvec4 size = values.lightGridSize;
    const uint32_t voxels = size.x * size.y * size.z;
    if (voxels == 0) {
        throw std::invalid_argument("TreeLightVolume: configure the light grid of the settings first");
    }
    const TreeCapacity& capacity = generator.getCapacity();

    auto densityBuffer = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(
        vulkanContext, voxels, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    densityBuffer->setName("TreeLightDensity");
    densityBuffer->setRecordToZero(true);
    light = std::make_shared<BufferElement<VulkanBuffer<glm::vec2>>>(vulkanContext, voxels);
    light->setName("TreeLight");

    density =
        std::make_shared<GeneralComputation<glm::uvec2>>(vulkanContext, "shaders/tree/tree_light_density.comp.spv");
    density->setName("TreeLightDensity");
    generator.getSegments().connectTo(*density, 0);
    generator.getLeaves().connectTo(*density, 1);
    generator.getCounters().connectTo(*density, 2);
    generator.getParamsRef().connectTo(*density, 3);
    density->setInput(settings, 4);
    density->setInput(densityBuffer, 5);
    density->setGroupCountX((capacity.segments + capacity.leaves + 255) / 256);
    density->setPushConstants({glm::uvec2(capacity.segments, capacity.leaves)});

    march = std::make_shared<GeneralComputation<>>(vulkanContext, "shaders/tree/tree_light_march.comp.spv");
    march->setName("TreeLightMarch");
    march->setInput(density, 0, 4);
    march->setInput(density, 1, 5);
    march->setInput(light, 2);
    march->setGroupCountX((voxels + 63) / 64);
}

std::vector<glm::vec2> TreeLightVolume::readLight(uint32_t pathId) {
    std::vector<glm::vec2> values(light->getBuffer(pathId).getSize());
    light->getBuffer(pathId).memcopyTo(values);
    return values;
}

} // namespace klartraum
