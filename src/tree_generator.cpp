// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/tree/tree_generator.hpp"

#include <algorithm>
#include <cmath>
#include <string>

namespace klartraum {

namespace {

uint32_t withMargin(size_t count, float margin) {
    return std::max<uint32_t>(1u, uint32_t(std::ceil(float(count) * margin)));
}

} // namespace

TreeCapacity TreeCapacity::fromReference(const std::vector<TreeTypeParams>& types,
                                         const std::vector<TreeInstance>& instances, const TreeFrame& frame,
                                         float margin) {
    const TreeRecords reference = generateTreesReference(types, instances, frame);
    TreeCapacity capacity;
    for (int level = 0; level < TREE_MAX_LEVELS; ++level) {
        capacity.stems[level] = withMargin(reference.stems[level].size(), margin);
    }
    capacity.segments = withMargin(reference.segments.size(), margin);
    capacity.leaves = withMargin(reference.leaves.size(), margin);
    return capacity;
}

TreeGenerator::TreeGenerator(VulkanContext& vulkanContext, std::vector<TreeTypeParams> types,
                             std::vector<TreeInstance> instanceList, const TreeCapacity& capacity,
                             const TreeFrame& frameValues)
    : capacity(capacity) {
    uint32_t stemTotal = 0;
    for (int level = 0; level < TREE_MAX_LEVELS; ++level) {
        push.stemOffset[level] = stemTotal;
        push.stemCapacity[level] = capacity.stems[level];
        stemTotal += capacity.stems[level];
    }
    push.instanceCount = uint32_t(instanceList.size());
    push.segmentCapacity = capacity.segments;
    push.leafCapacity = capacity.leaves;

    params = std::make_shared<HostValues<TreeTypeParams>>(vulkanContext, std::move(types));
    params->setName("TreeParams");
    instances = std::make_shared<HostValues<TreeInstance>>(vulkanContext, std::move(instanceList));
    instances->setName("TreeInstances");
    frame = std::make_shared<HostValues<TreeFrame>>(vulkanContext, std::vector<TreeFrame>{frameValues});
    frame->setName("TreeFrame");

    stems = std::make_shared<BufferElement<VulkanBuffer<StemRecord>>>(vulkanContext, stemTotal);
    stems->setName("TreeStems");
    segments = std::make_shared<BufferElement<VulkanBuffer<SegmentRecord>>>(vulkanContext, capacity.segments);
    segments->setName("TreeSegments");
    leaves = std::make_shared<BufferElement<VulkanBuffer<LeafRecord>>>(vulkanContext, capacity.leaves);
    leaves->setName("TreeLeaves");
    counters = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanContext, kTreeCounterCount);
    counters->setName("TreeCounters");
    counters->setRecordToZero(true);

    const VkBufferUsageFlags argsUsage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    for (int level = 0; level < TREE_MAX_LEVELS; ++level) {
        args[level] = std::make_shared<DispatchIndirectCommandBufferElement>(vulkanContext, 1, argsUsage,
                                                                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        args[level]->setName("TreeArgs" + std::to_string(level));
    }

    // Each pass takes the buffers through the previous one, so the passes run in order.
    auto addPass = [&](const std::string& shader, const std::string& name, uint32_t level,
                       const std::shared_ptr<DispatchIndirectCommandBufferElement>& argsBuffer) {
        auto pass = std::make_shared<GeneralComputation<TreeGenPush>>(vulkanContext, shader);
        pass->setName(name);
        if (passes.empty()) {
            pass->setInput(params, 0);
            pass->setInput(instances, 1);
            pass->setInput(frame, 2);
            pass->setInput(stems, 3);
            pass->setInput(segments, 4);
            pass->setInput(leaves, 5);
            pass->setInput(counters, 6);
        } else {
            for (int i = 0; i < 7; ++i) {
                pass->setInput(passes.back(), i, i);
            }
        }
        pass->setInput(argsBuffer, 7);
        TreeGenPush levelPush = push;
        levelPush.level = level;
        pass->setPushConstants({levelPush});
        passes.push_back(pass);
        return pass;
    };

    auto roots = addPass("shaders/tree/tree_roots.comp.spv", "TreeRoots", 0, args[0]);
    roots->setGroupCountX((push.instanceCount + 63) / 64);
    for (uint32_t level = 0; level < TREE_MAX_LEVELS; ++level) {
        addPass("shaders/tree/tree_args.comp.spv", "TreeArgs" + std::to_string(level), level, args[level]);
        auto pass =
            addPass("shaders/tree/tree_level.comp.spv", "TreeLevel" + std::to_string(level), level, args[level]);
        pass->setDynamicGroupDispatchParams(args[level]);
    }
    lastPass = passes.back();
    outputElements[0] = lastPass;
}

std::vector<uint32_t> TreeGenerator::readCounters(uint32_t pathId) {
    std::vector<uint32_t> values(kTreeCounterCount, 0u);
    counters->getBuffer(pathId).memcopyTo(values);
    return values;
}

TreeRecords TreeGenerator::readRecords(uint32_t pathId) {
    const std::vector<uint32_t> count = readCounters(pathId);
    TreeRecords records;
    std::vector<StemRecord> allStems(stems->getBuffer(pathId).getSize());
    stems->getBuffer(pathId).memcopyTo(allStems);
    for (int level = 0; level < TREE_MAX_LEVELS; ++level) {
        const uint32_t n = std::min(count[level], capacity.stems[level]);
        auto first = allStems.begin() + push.stemOffset[level];
        records.stems[level].assign(first, first + n);
    }
    records.segments.resize(std::min(count[kTreeCounterSegments], capacity.segments));
    segments->getBuffer(pathId).memcopyTo(records.segments);
    records.leaves.resize(std::min(count[kTreeCounterLeaves], capacity.leaves));
    leaves->getBuffer(pathId).memcopyTo(records.leaves);
    return records;
}

} // namespace klartraum
