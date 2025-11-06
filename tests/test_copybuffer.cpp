#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "klartraum/computegraph/bufferelement.hpp"
#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/copybuffer.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/vulkan_buffer.hpp"
#include "klartraum/vulkan_helpers.hpp"

using namespace klartraum;

class CopyBufferTest : public ::testing::Test {
protected:
    void SetUp() override {
        frontend = std::make_unique<HeadlessFrontend>();
        vulkanContext = &frontend->getKlartraumEngine().getVulkanContext();
    }

    std::unique_ptr<HeadlessFrontend> frontend;
    VulkanContext* vulkanContext;
};

TEST_F(CopyBufferTest, BasicCopyBufferTest) {
    const uint32_t numberElements = 10;
    const uint32_t numberPaths = 1; // simple test for single path, TODO add test for multiple paths

    // Create source buffer with test data
    VkBufferUsageFlags srcFlags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    auto sourceBuffer = std::make_shared<BufferElement<VulkanBuffer<float>>>(*vulkanContext, numberElements, srcFlags);

    // Create destination buffer
    auto destBuffer = std::make_shared<BufferElement<VulkanBuffer<float>>>(*vulkanContext, numberElements);

    // Create CopyBuffer element
    auto copyBuffer = std::make_shared<CopyBuffer>(*vulkanContext);
    copyBuffer->setInput<0>(sourceBuffer);
    copyBuffer->setInput<1>(destBuffer);

    // Set up test data
    std::vector<float> testData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};

    // Create and compile compute graph
    ComputeGraph computeGraph(*vulkanContext, numberPaths);

    // Compile the graph
    computeGraph.compileFrom(copyBuffer);

    // Copy test data to source buffer
    sourceBuffer->getBuffer(0).memcopyFrom(testData);

    // Execute the copy operation
    computeGraph.submitAndWait(vulkanContext->getGraphicsQueue(), 0);

    // Verify the copy worked by reading from destination buffer
    std::vector<float> resultData(numberElements);
    destBuffer->getBuffer(0).memcopyTo(resultData);

    // Verify the data matches
    for (size_t i = 0; i < testData.size(); ++i) {
        EXPECT_FLOAT_EQ(testData[i], resultData[i])
            << "Mismatch at index " << i << ": expected " << testData[i]
            << ", got " << resultData[i];
    }
}
