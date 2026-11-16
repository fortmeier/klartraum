// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/tree/tree_light_volume.hpp"

#include <algorithm>
#include <array>
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
                                 std::shared_ptr<HostValues<TreeRenderSettings>> settings,
                                 std::vector<float> staticDepth, TreeMovingOccluders moving) {
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
    light = std::make_shared<BufferElement<VulkanBuffer<glm::vec4>>>(vulkanContext, voxels);
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
    if (staticDepth.size() != voxels) {
        if (!staticDepth.empty()) {
            throw std::invalid_argument("TreeLightVolume: staticDepth must have one value per voxel");
        }
        staticDepth.assign(voxels, 0.0f);
    }
    auto staticBuffer = std::make_shared<BufferElementSinglePath<VulkanBuffer<float>>>(
        vulkanContext, voxels, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    staticBuffer->setName("TreeStaticDepth");
    staticBuffer->getBuffer().memcopyFrom(staticDepth);
    march->setInput(staticBuffer, 3);

    // The moving occluders' grid, zeroed every frame; without moving occluders it stays zero.
    auto movingBuffer = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(
        vulkanContext, voxels, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    movingBuffer->setName("TreeMovingDepth");
    movingBuffer->setRecordToZero(true);
    movingTransforms = std::make_shared<HostValues<glm::mat4>>(
        vulkanContext, std::vector<glm::mat4>(std::max(moving.objectCount, 1u), glm::mat4(1.0f)));
    movingTransforms->setName("TreeMovingTransforms");
    if (moving.surfels.empty()) {
        march->setInput(movingBuffer, 4);
    } else {
        if (moving.objects.size() != moving.surfels.size()) {
            throw std::invalid_argument("TreeLightVolume: one object index per moving occluder sample");
        }
        for (const uint32_t object : moving.objects) {
            if (object >= moving.objectCount) {
                throw std::invalid_argument("TreeLightVolume: moving occluder object index out of range");
            }
        }
        const auto count = uint32_t(moving.surfels.size());
        auto samples = std::make_shared<BufferElementSinglePath<VulkanBuffer<glm::vec4>>>(
            vulkanContext, count, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        samples->setName("TreeMovingSamples");
        samples->getBuffer().memcopyFrom(moving.surfels);
        auto objects = std::make_shared<BufferElementSinglePath<VulkanBuffer<uint32_t>>>(
            vulkanContext, count, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        objects->setName("TreeMovingObjects");
        objects->getBuffer().memcopyFrom(moving.objects);
        auto movers = std::make_shared<GeneralComputation<TreeMoversPush>>(vulkanContext,
                                                                           "shaders/tree/tree_light_movers.comp.spv");
        movers->setName("TreeLightMovers");
        movers->setInput(samples, 0);
        movers->setInput(objects, 1);
        movers->setInput(movingTransforms, 2);
        movers->setInput(settings, 3);
        movers->setInput(movingBuffer, 4);
        movers->setGroupCountX((count + 255) / 256);
        movers->setPushConstants({TreeMoversPush{count, kPerSample, 0.0f, 0u}});
        march->setInput(movers, 4, 4);
        movingGrid = {movers, 4};
    }
    if (!movingGrid.element) {
        movingGrid = {movingBuffer, -1};
    }
    this->settings = settings;
    this->vulkanContext = &vulkanContext;
    movingObjectCount = moving.objectCount;
    march->setGroupCountX((voxels + 63) / 64);
}

void TreeLightVolume::setMovingTransform(uint32_t object, const glm::mat4& worldFromObject) {
    movingTransforms->set(object, worldFromObject);
}

void TreeLightVolume::addLiveOccluder(const BufferRef& vertices, uint32_t count, float area, uint32_t object,
                                      const ComputeGraphElementPtr& after) {
    if (object >= movingObjectCount) {
        throw std::invalid_argument("TreeLightVolume: live occluder object index out of range");
    }
    // The pass's per-sample objects binding needs a buffer; the push constants give the object instead.
    auto unused = std::make_shared<BufferElementSinglePath<VulkanBuffer<uint32_t>>>(*vulkanContext, 1);
    unused->setName("TreeLiveOccluderObjects");
    auto movers =
        std::make_shared<GeneralComputation<TreeMoversPush>>(*vulkanContext, "shaders/tree/tree_light_movers.comp.spv");
    movers->setName("TreeLightLiveOccluder");
    vertices.connectTo(*movers, 0);
    movers->setInput(unused, 1);
    movers->setInput(movingTransforms, 2);
    movers->setInput(settings, 3);
    movingGrid.connectTo(*movers, 4);
    if (after) {
        movers->addDependency(after);
    }
    movers->setGroupCountX((count + 255) / 256);
    movers->setPushConstants({TreeMoversPush{count, object, area, 0u}});
    march->setInput(movers, 4, 4);
    movingGrid = {movers, 4};
}

std::vector<float> treeStaticOcclusion(const TreeRenderSettings& settings, const std::vector<TreeSceneVertex>& surfels,
                                       const std::vector<Gaussian3D>& gaussians, float opaqueDepth) {
    const glm::uvec3 size(settings.lightGridSize);
    std::vector<float> depth(size_t(size.x) * size.y * size.z, 0.0f);
    const float voxel = settings.lightGridMin.w;
    const float perArea = opaqueDepth / (voxel * voxel);
    auto add = [&](glm::vec3 p, float area) {
        const glm::ivec3 c = glm::ivec3(glm::floor((p - glm::vec3(settings.lightGridMin)) / voxel));
        if (glm::any(glm::lessThan(c, glm::ivec3(0))) || glm::any(glm::greaterThanEqual(c, glm::ivec3(size)))) {
            return;
        }
        float& d = depth[tree::treeLightIndex(uint32_t(c.x), uint32_t(c.y), uint32_t(c.z), settings.lightGridSize)];
        d = std::min(d + perArea * area, 2.0f * opaqueDepth);
    };
    for (const TreeSceneVertex& s : surfels) {
        add(glm::vec3(s.pos), s.pos.w * s.pos.w);
    }
    for (const Gaussian3D& g : gaussians) {
        std::array<float, 3> scale = g.scale;
        std::sort(scale.begin(), scale.end());
        add(glm::vec3(g.position[0], g.position[1], g.position[2]), g.alpha * 3.14159265f * scale[1] * scale[2]);
    }
    return depth;
}

std::vector<glm::vec4> treeOccluderSamples(const std::vector<TreeSceneVertex>& surfels,
                                           const std::vector<Gaussian3D>& gaussians) {
    std::vector<glm::vec4> samples;
    samples.reserve(surfels.size() + gaussians.size());
    for (const TreeSceneVertex& s : surfels) {
        samples.emplace_back(glm::vec3(s.pos), s.pos.w * s.pos.w);
    }
    for (const Gaussian3D& g : gaussians) {
        std::array<float, 3> scale = g.scale;
        std::sort(scale.begin(), scale.end());
        samples.emplace_back(g.position[0], g.position[1], g.position[2], g.alpha * 3.14159265f * scale[1] * scale[2]);
    }
    return samples;
}

std::vector<glm::vec4> TreeLightVolume::readLight(uint32_t pathId) {
    std::vector<glm::vec4> values(light->getBuffer(pathId).getSize());
    light->getBuffer(pathId).memcopyTo(values);
    return values;
}

} // namespace klartraum
