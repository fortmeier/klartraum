/**
 * TESTS:
 * - DrawIndirectCommandBufferElement can be compiled and submitted as a graph leaf
 *   holding a VkDrawIndirectCommand
 * - setRecordToZeroRange resets only the targeted byte range (instanceCount) on each
 *   submission while leaving the rest of the struct (vertexCount) untouched
 * - setRecordToFill overwrites the whole buffer with a repeating 32-bit sentinel
 *   pattern (0xFFFFFFFF) on each submission, replacing previously seeded data
 **/
#include <gtest/gtest.h>

#include <cstddef>

#include "klartraum/headless_frontend.hpp"
#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/bufferelement.hpp"
#include "klartraum/computegraph/buffertransformation.hpp"
#include "klartraum/vulkan_buffer.hpp"

using namespace klartraum;

TEST(DrawIndirectCommandBufferElement, partialResetPreservesVertexCount) {
    klartraum::HeadlessFrontend frontend;

    auto& core = frontend.getKlartraumEngine();
    auto& vulkanContext = core.getVulkanContext();

    auto drawArgs = std::make_shared<DrawIndirectCommandBufferElement>(vulkanContext, 1,
        VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    // Reset only instanceCount each frame; vertexCount must survive across submissions.
    drawArgs->setRecordToZeroRange(offsetof(VkDrawIndirectCommand, instanceCount), sizeof(uint32_t));

    auto computegraph = ComputeGraph(vulkanContext, 1);
    computegraph.compileFrom(drawArgs);

    VkDrawIndirectCommand initial{4, 0, 0, 0};
    drawArgs->getBuffer(0).memcopyFrom(&initial, 1);

    // Simulate a previous frame's compute pass having bumped instanceCount.
    VkDrawIndirectCommand afterCompute{4, 42, 0, 0};
    drawArgs->getBuffer(0).memcopyFrom(&afterCompute, 1);

    computegraph.submitAndWait(vulkanContext.getGraphicsQueue(), 0);

    std::vector<VkDrawIndirectCommand> result(1);
    drawArgs->getBuffer(0).memcopyTo(result);

    EXPECT_EQ(result[0].vertexCount, 4u);
    EXPECT_EQ(result[0].instanceCount, 0u);
    EXPECT_EQ(result[0].firstVertex, 0u);
    EXPECT_EQ(result[0].firstInstance, 0u);
}

TEST(BufferElement, fillOverwritesBufferWithSentinelPatternEachFrame) {
    klartraum::HeadlessFrontend frontend;

    auto& core = frontend.getKlartraumEngine();
    auto& vulkanContext = core.getVulkanContext();

    const uint32_t numElements = 8;
    auto keys = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanContext, numElements,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    // Sentinel guaranteed to sort after any encoded depth key (guide §5 Stage B
    // sizing scheme): each frame the whole keys buffer is reset to it before the
    // compute stage writes real keys for the visible splats.
    keys->setRecordToFill(0xFFFFFFFFu);

    auto computegraph = ComputeGraph(vulkanContext, 1);
    computegraph.compileFrom(keys);

    // Simulate a previous frame having written real (smaller) key values.
    std::vector<uint32_t> previousFrameData(numElements, 0x12345678u);
    keys->getBuffer(0).memcopyFrom(previousFrameData);

    computegraph.submitAndWait(vulkanContext.getGraphicsQueue(), 0);

    std::vector<uint32_t> result(numElements);
    keys->getBuffer(0).memcopyTo(result);

    for (uint32_t i = 0; i < numElements; i++) {
        EXPECT_EQ(result[i], 0xFFFFFFFFu);
    }
}
