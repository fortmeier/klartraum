// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/tree/tree_stem_renderer.hpp"

namespace klartraum {

namespace {

struct StemDrawPush {
    uint32_t bucket;
    uint32_t capacity;
};

} // namespace

TreeStemRenderer::TreeStemRenderer(VulkanContext& vulkanContext, TreeGenerator& generator,
                                   std::shared_ptr<CameraUboType> camera,
                                   std::shared_ptr<HostValues<TreeRenderSettings>> settings, VkExtent2D extent) {
    const uint32_t capacity = generator.getCapacity().segments;

    lists = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanContext, capacity * TREE_STEM_BUCKETS,
                                                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    lists->setName("TreeStemLists");
    drawArgs = std::make_shared<DrawIndirectCommandBufferElement>(
        vulkanContext, TREE_STEM_BUCKETS,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    drawArgs->setName("TreeStemDrawArgs");
    drawArgs->setRecordToZero(true);
    stats = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanContext, 4);
    stats->setName("TreeStemStats");
    stats->setRecordToZero(true);

    classify =
        std::make_shared<GeneralComputation<uint32_t>>(vulkanContext, "shaders/tree/tree_stem_classify.comp.spv");
    classify->setName("TreeStemClassify");
    generator.getSegments().connectTo(*classify, 0);
    generator.getCounters().connectTo(*classify, 1);
    classify->setInput(camera, 2);
    classify->setInput(settings, 3);
    classify->setInput(lists, 4);
    classify->setInput(drawArgs, 5);
    classify->setInput(stats, 6);
    classify->setInput(generator.getParamsRef().element, 7, generator.getParamsRef().slot);
    classify->setGroupCountX((capacity + 63) / 64);
    classify->setPushConstants({capacity});

    StorageDrawConfig config;
    config.vertexShader = "shaders/tree/tree_stem.vert.spv";
    config.fragmentShader = "shaders/tree/tree_stem.frag.spv";
    config.extent = extent;
    config.pushConstantSize = sizeof(StemDrawPush);
    draw = std::make_shared<StorageDrawComponent>(
        config, std::vector<BufferRef>{{classify, 0}, {classify, 4}, {classify, 7}, {classify, 3}});
    for (uint32_t bucket = 0; bucket < TREE_STEM_BUCKETS; ++bucket) {
        StorageDrawComponent::Draw d;
        d.indirect = {classify, 5};
        d.offset = bucket * sizeof(VkDrawIndirectCommand);
        d.pushConstants = StorageDrawComponent::bytes(StemDrawPush{bucket, capacity});
        draw->addDraw(std::move(d));
    }
}

std::vector<uint32_t> TreeStemRenderer::readStats(uint32_t pathId) {
    std::vector<uint32_t> values(4, 0u);
    stats->getBuffer(pathId).memcopyTo(values);
    return values;
}

} // namespace klartraum
