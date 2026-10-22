// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/tree/tree_moss_lod.hpp"

#include <algorithm>
#include <stdexcept>

namespace klartraum {

TreeMossLod::TreeMossLod(VulkanContext& vulkanContext, TreeGaussians& target, const std::vector<MossPatch>& patches,
                         std::vector<TreeLodInstance> instances, std::shared_ptr<CameraUboType> camera,
                         std::shared_ptr<HostValues<TreeRenderSettings>> settings, BufferRef light,
                         glm::vec4 boundaries, uint32_t poolSlots) {
    if (patches.empty() || instances.empty()) {
        throw std::invalid_argument("TreeMossLod: needs patches and placed copies");
    }
    const uint32_t capacity = poolSlots > 0 ? std::min(poolSlots, target.getExtraSlots()) : target.getExtraSlots();
    auto instanceBuffer = std::make_shared<BufferElementSinglePath<VulkanBuffer<TreeLodInstance>>>(
        vulkanContext, uint32_t(instances.size()));
    instanceBuffer->setName("TreeLodInstances");
    instanceBuffer->getBuffer().memcopyFrom(instances);
    build(vulkanContext, target.getGaussians(), patches, {instanceBuffer, -1}, uint32_t(instances.size()), camera,
          settings, light, boundaries, capacity, target.getExtraOffset());
    for (const TreeLodInstance& inst : instances) {
        fullCount += table.at(inst.ids.x).count[0];
    }
}

TreeMossLod::TreeMossLod(VulkanContext& vulkanContext, const GaussianSoABuffers& source,
                         const std::vector<MossPatch>& patches, BufferRef instances, uint32_t instanceCount,
                         std::shared_ptr<CameraUboType> camera,
                         std::shared_ptr<HostValues<TreeRenderSettings>> settings, BufferRef light,
                         glm::vec4 boundaries, uint32_t poolSlots, uint32_t poolOffset) {
    if (patches.empty() || instanceCount == 0) {
        throw std::invalid_argument("TreeMossLod: needs patches and placed copies");
    }
    build(vulkanContext, source, patches, instances, instanceCount, camera, settings, light, boundaries, poolSlots,
          poolOffset);
}

void TreeMossLod::build(VulkanContext& vulkanContext, const GaussianSoABuffers& source,
                        const std::vector<MossPatch>& patches, BufferRef instanceBuffer, uint32_t instanceCount,
                        std::shared_ptr<CameraUboType> camera, std::shared_ptr<HostValues<TreeRenderSettings>> settings,
                        BufferRef light, glm::vec4 boundaries, uint32_t capacity, uint32_t offset) {
    // The library: every patch's four levels, one after the other.
    std::vector<TreeLodSplat> library;
    for (const MossPatch& patch : patches) {
        const GaussianLods lods = buildGaussianLods(patch);
        TreeLodPatch entry{};
        float radius = 0.0f;
        for (const Gaussian3D& g : patch) {
            radius = std::max(radius, glm::length(glm::vec2(g.position[0], g.position[2])));
        }
        entry.bounds = glm::vec4(radius, 0.0f, 0.0f, 0.0f);
        for (int level = 0; level < 4; ++level) {
            entry.offset[level] = uint32_t(library.size());
            entry.count[level] = uint32_t(lods.levels[level].size());
            for (const Gaussian3D& g : lods.levels[level]) {
                TreeLodSplat splat{};
                splat.pos = glm::vec4(g.position[0], g.position[1], g.position[2], 0.0f);
                splat.rot = glm::vec4(g.rotation[0], g.rotation[1], g.rotation[2], g.rotation[3]);
                splat.scale = glm::vec4(g.scale[0], g.scale[1], g.scale[2], 0.0f);
                splat.color = glm::vec4(g.color[0], g.color[1], g.color[2], g.alpha);
                library.push_back(splat);
            }
        }
        table.push_back(entry);
    }
    auto upload = [&](const auto& data, const char* name) {
        using T = typename std::decay_t<decltype(data)>::value_type;
        auto element = std::make_shared<BufferElementSinglePath<VulkanBuffer<T>>>(vulkanContext, uint32_t(data.size()));
        element->setName(name);
        element->getBuffer().memcopyFrom(data);
        return element;
    };
    auto libraryBuffer = upload(library, "TreeLodLibrary");
    auto patchBuffer = upload(table, "TreeLodPatches");
    auto records = std::make_shared<BufferElement<VulkanBuffer<glm::uvec4>>>(
        vulkanContext, instanceCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    records->setName("TreeLodRecords");
    counter = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanContext, 4);
    counter->setName("TreeLodCounter");
    counter->setRecordToZero(true);

    clear = std::make_shared<GeneralComputation<glm::uvec2>>(vulkanContext, "shaders/tree/tree_lod_clear.comp.spv");
    clear->setName("TreeLodClear");
    source.colAlpha.connectTo(*clear, 0);
    clear->setGroupCountX((capacity + 255) / 256);
    clear->setPushConstants({glm::uvec2(offset, capacity)});

    select =
        std::make_shared<GeneralComputation<TreeLodSelectPush>>(vulkanContext, "shaders/tree/tree_lod_select.comp.spv");
    select->setName("TreeLodSelect");
    instanceBuffer.connectTo(*select, 0);
    select->setInput(patchBuffer, 1);
    select->setInput(records, 2);
    select->setInput(counter, 3);
    select->setInput(camera, 4);
    select->setInput(settings, 5);
    select->setGroupCountX((instanceCount + 63) / 64);
    TreeLodSelectPush push;
    push.instanceCount = instanceCount;
    push.capacity = capacity;
    push.boundaries = boundaries;
    select->setPushConstants({push});

    write = std::make_shared<GeneralComputation<uint32_t>>(vulkanContext, "shaders/tree/tree_lod_write.comp.spv");
    write->setName("TreeLodWrite");
    source.pos.connectTo(*write, 0);
    source.rot.connectTo(*write, 1);
    source.scale.connectTo(*write, 2);
    write->setInput(clear, 3, 0);
    write->setInput(libraryBuffer, 4);
    write->setInput(select, 5, 1);
    write->setInput(select, 6, 0);
    write->setInput(select, 7, 2);
    write->setInput(settings, 8);
    const BufferRef lightRef = bufferOrPlaceholder(vulkanContext, light);
    write->setInput(lightRef.element, 9, lightRef.slot);
    write->setGroupCountX(instanceCount);
    write->setPushConstants({offset});

    gaussians = source;
    gaussians.pos = BufferRef{write, 0};
    gaussians.rot = BufferRef{write, 1};
    gaussians.scale = BufferRef{write, 2};
    gaussians.colAlpha = BufferRef{write, 3};
}

std::vector<uint32_t> TreeMossLod::readCounter(uint32_t pathId) {
    std::vector<uint32_t> values(4, 0u);
    counter->getBuffer(pathId).memcopyTo(values);
    return values;
}

} // namespace klartraum
