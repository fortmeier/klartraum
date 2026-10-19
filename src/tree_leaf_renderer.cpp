// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/tree/tree_leaf_renderer.hpp"

#include <algorithm>

namespace klartraum {

namespace {

constexpr uint32_t kLeafBuckets = 2;

struct LeafDrawPush {
    uint32_t bucket;
    uint32_t capacity;
    uint32_t lobes;
};

} // namespace

TreeLeafRenderer::TreeLeafRenderer(VulkanContext& vulkanContext, TreeGenerator& generator,
                                   const std::vector<TreeTypeParams>& types, std::shared_ptr<CameraUboType> camera,
                                   std::shared_ptr<HostValues<TreeRenderSettings>> settings, VkExtent2D extent,
                                   TreeRenderInputs inputs) {
    const uint32_t capacity = generator.getCapacity().leaves;
    // Every leaf is drawn with as many lobes as the type with the most; the others collapse.
    uint32_t lobes = 1;
    for (const TreeTypeParams& type : types) {
        lobes = std::max(lobes, uint32_t(std::clamp(type.leafLobes.x, 1.0f, float(kMaxLobes))));
    }

    lists = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanContext, capacity * kLeafBuckets,
                                                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    lists->setName("TreeLeafLists");
    drawArgs = std::make_shared<DrawIndirectCommandBufferElement>(
        vulkanContext, kLeafBuckets,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    drawArgs->setName("TreeLeafDrawArgs");
    drawArgs->setRecordToZero(true);
    stats = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanContext, 4);
    stats->setName("TreeLeafStats");
    stats->setRecordToZero(true);

    classify =
        std::make_shared<GeneralComputation<glm::uvec2>>(vulkanContext, "shaders/tree/tree_leaf_classify.comp.spv");
    classify->setName("TreeLeafClassify");
    generator.getLeaves().connectTo(*classify, 0);
    generator.getCounters().connectTo(*classify, 1);
    classify->setInput(camera, 2);
    classify->setInput(settings, 3);
    classify->setInput(lists, 4);
    classify->setInput(drawArgs, 5);
    classify->setInput(stats, 6);
    generator.getParamsRef().connectTo(*classify, 7);
    classify->setGroupCountX((capacity + 63) / 64);
    classify->setPushConstants({glm::uvec2(capacity, lobes)});

    StorageDrawConfig config;
    config.vertexShader = "shaders/tree/tree_leaf.vert.spv";
    config.fragmentShader = "shaders/tree/tree_leaf.frag.spv";
    config.extent = extent;
    config.pushConstantSize = sizeof(LeafDrawPush);
    draw = std::make_shared<StorageDrawComponent>(
        config, std::vector<BufferRef>{{classify, 0},
                                       {classify, 4},
                                       {classify, 7},
                                       {classify, 3},
                                       bufferOrPlaceholder(vulkanContext, inputs.sceneDepth),
                                       bufferOrPlaceholder(vulkanContext, inputs.light)});
    for (uint32_t bucket = 0; bucket < kLeafBuckets; ++bucket) {
        StorageDrawComponent::Draw d;
        d.indirect = {classify, 5};
        d.offset = bucket * sizeof(VkDrawIndirectCommand);
        d.pushConstants = StorageDrawComponent::bytes(LeafDrawPush{bucket, capacity, lobes});
        draw->addDraw(std::move(d));
    }
}

std::vector<uint32_t> TreeLeafRenderer::readStats(uint32_t pathId) {
    std::vector<uint32_t> values(4, 0u);
    stats->getBuffer(pathId).memcopyTo(values);
    return values;
}

} // namespace klartraum
