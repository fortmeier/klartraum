// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - memcopy: data copied into a VulkanBuffer and back to the host is unchanged
 * - deviceLocalUploadsAndReadbackAreBounded: vector, typed and byte uploads preserve
 *   values and untouched tails; short and empty transfers respect buffer capacity
 * - deviceLocalZeroClearsAllElements: GPU filling clears an initialized device-local buffer
 * - movedDeviceLocalBufferRemainsUsable: moving preserves storage properties and transfer behavior
 * - deviceLocalBufferElementKeepsPathsIndependent: device-local graph buffers retain separate path data
 * - deviceLocalBuffersFeedGpuCompute: staged input reaches a real dispatch and its result can be read back
 * - copyFromPreservesTailAndSource: prefix copies preserve the tail and source, clamp to
 *   destination capacity and leave zero-length copies unchanged
 * - copyStaticPrefixToIndependentPaths: each path receives a static prefix while retaining its own tail
 **/
#include <gtest/gtest.h>

#include <utility>
#include <vector>

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/generalcomputation.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/vulkan_buffer.hpp"

TEST(VulkanBuffer, memcopy) {
    klartraum::HeadlessFrontend frontend;

    auto& core = frontend.getKlartraumEngine();
    auto& vulkanContext = core.getVulkanContext();
    auto& device = vulkanContext.getDevice();

    klartraum::VulkanBuffer<float> buffer(vulkanContext, 7);
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    buffer.memcopyFrom(data);
    std::vector<float> data2(7);
    buffer.memcopyTo(data2);
    for (int i = 0; i < 7; i++) {
        EXPECT_EQ(data[i], data2[i]);
    }
    return;
}

TEST(VulkanBuffer, deviceLocalUploadsAndReadbackAreBounded) {
    klartraum::HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    klartraum::VulkanBuffer<uint32_t> buffer(vc, 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    EXPECT_NE(buffer.getMemoryProperties() & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0u);
    buffer.memcopyFrom(std::vector<uint32_t>{11, 22, 33, 44, 55, 66});
    std::vector<uint32_t> result(4);
    buffer.memcopyTo(result);
    EXPECT_EQ(result, (std::vector<uint32_t>{11, 22, 33, 44}));

    const uint32_t typed[] = {101, 202};
    buffer.memcopyFrom(typed, 2);
    buffer.memcopyTo(result);
    EXPECT_EQ(result, (std::vector<uint32_t>{101, 202, 33, 44}));
    const uint32_t bytes[] = {31, 32};
    buffer.memcopyFrom(reinterpret_cast<const char*>(bytes), sizeof(bytes));
    buffer.memcopyFrom(std::vector<uint32_t>{});
    buffer.memcopyTo(result);
    EXPECT_EQ(result, (std::vector<uint32_t>{31, 32, 33, 44}));
    std::vector<uint32_t> prefix(2);
    buffer.memcopyTo(prefix);
    EXPECT_EQ(prefix, (std::vector<uint32_t>{31, 32}));
    std::vector<uint32_t> empty;
    buffer.memcopyTo(empty);
    EXPECT_TRUE(empty.empty());
}

TEST(VulkanBuffer, deviceLocalZeroClearsAllElements) {
    klartraum::HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    klartraum::VulkanBuffer<uint32_t> buffer(vc, 5, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    buffer.memcopyFrom(std::vector<uint32_t>{1, 2, 3, 4, 5});
    buffer.zero();
    std::vector<uint32_t> result(5, 99);
    buffer.memcopyTo(result);
    EXPECT_EQ(result, std::vector<uint32_t>(5, 0));
}

TEST(VulkanBuffer, movedDeviceLocalBufferRemainsUsable) {
    klartraum::HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    klartraum::VulkanBuffer<uint32_t> source(vc, 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    source.memcopyFrom(std::vector<uint32_t>{1, 2, 3, 4});
    const auto properties = source.getMemoryProperties();
    klartraum::VulkanBuffer<uint32_t> moved(std::move(source));
    EXPECT_EQ(moved.getMemoryProperties(), properties);
    std::vector<uint32_t> result(4);
    moved.memcopyTo(result);
    EXPECT_EQ(result, (std::vector<uint32_t>{1, 2, 3, 4}));
    moved.memcopyFrom(std::vector<uint32_t>{5, 6, 7, 8});
    moved.memcopyTo(result);
    EXPECT_EQ(result, (std::vector<uint32_t>{5, 6, 7, 8}));
}

TEST(VulkanBuffer, deviceLocalBufferElementKeepsPathsIndependent) {
    klartraum::HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    auto buffers = std::make_shared<klartraum::BufferElement<klartraum::VulkanBuffer<uint32_t>>>(
        vc, 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    klartraum::ComputeGraph graph(vc, 2);
    graph.compileFrom(buffers);
    for (uint32_t path = 0; path < 2; ++path) {
        EXPECT_NE(buffers->getBuffer(path).getMemoryProperties() & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0u);
        buffers->getBuffer(path).memcopyFrom(std::vector<uint32_t>(4, path + 10));
    }
    EXPECT_NE(buffers->getVkBuffer(0), buffers->getVkBuffer(1));
    buffers->getBuffer(1).zero();
    std::vector<uint32_t> result(4);
    buffers->getBuffer(0).memcopyTo(result);
    EXPECT_EQ(result, std::vector<uint32_t>(4, 10));
    buffers->getBuffer(1).memcopyTo(result);
    EXPECT_EQ(result, std::vector<uint32_t>(4, 0));
}

TEST(VulkanBuffer, deviceLocalBuffersFeedGpuCompute) {
    klartraum::HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    using Buffer = klartraum::BufferElement<klartraum::VulkanBuffer<float>>;
    auto input =
        std::make_shared<Buffer>(vc, 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    auto output =
        std::make_shared<Buffer>(vc, 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    auto factor = std::make_shared<Buffer>(vc, 4);
    auto multiply =
        std::make_shared<klartraum::GeneralComputation<>>(vc, "shaders/operator_multiply_scalar_element_wise.comp.spv");
    multiply->setInput(input, 0);
    multiply->setInput(factor, 1);
    multiply->setInput(output, 2);
    multiply->setGroupCountX(4);
    klartraum::ComputeGraph graph(vc, 1);
    graph.compileFrom(multiply);
    input->getBuffer(0).memcopyFrom(std::vector<float>{1, 2, 3, 4});
    factor->getBuffer(0).memcopyFrom(std::vector<float>{2, 3, 4, 5});
    graph.submitAndWait(vc.getGraphicsQueue(), 0);
    std::vector<float> result(4);
    output->getBuffer(0).memcopyTo(result);
    EXPECT_EQ(result, (std::vector<float>{2, 6, 12, 20}));
}

TEST(VulkanBuffer, copyFromPreservesTailAndSource) {
    klartraum::HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    klartraum::VulkanBuffer<uint32_t> source(vc, 6, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    klartraum::VulkanBuffer<uint32_t> destination(vc, 5, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    const std::vector<uint32_t> original{1, 2, 3, 4, 5, 6};
    source.memcopyFrom(original);
    destination.memcopyFrom(std::vector<uint32_t>(5, 99));
    destination.copyFrom(source.getBuffer(), 3);
    destination.copyFrom(VK_NULL_HANDLE, 0);
    std::vector<uint32_t> result(5);
    destination.memcopyTo(result);
    EXPECT_EQ(result, (std::vector<uint32_t>{1, 2, 3, 99, 99}));

    destination.copyFrom(source.getBuffer(), 8);
    destination.memcopyTo(result);
    EXPECT_EQ(result, (std::vector<uint32_t>{1, 2, 3, 4, 5}));
    std::vector<uint32_t> unchanged(6);
    source.memcopyTo(unchanged);
    EXPECT_EQ(unchanged, original);
}

TEST(VulkanBuffer, copyStaticPrefixToIndependentPaths) {
    klartraum::HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    klartraum::VulkanBuffer<uint32_t> source(vc, 3, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    source.memcopyFrom(std::vector<uint32_t>{1, 2, 3});
    auto outputs = std::make_shared<klartraum::BufferElement<klartraum::VulkanBuffer<uint32_t>>>(
        vc, 5, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    klartraum::ComputeGraph graph(vc, 2);
    graph.compileFrom(outputs);
    for (uint32_t path = 0; path < 2; ++path) {
        outputs->getBuffer(path).memcopyFrom(std::vector<uint32_t>(5, path + 10));
        outputs->getBuffer(path).copyFrom(source.getBuffer(), 3);
        std::vector<uint32_t> result(5);
        outputs->getBuffer(path).memcopyTo(result);
        EXPECT_EQ(result, (std::vector<uint32_t>{1, 2, 3, path + 10, path + 10}));
    }
    outputs->getBuffer(1).memcopyFrom(std::vector<uint32_t>{77});
    std::vector<uint32_t> result(5);
    outputs->getBuffer(0).memcopyTo(result);
    EXPECT_EQ(result, (std::vector<uint32_t>{1, 2, 3, 10, 10}));
    outputs->getBuffer(1).memcopyTo(result);
    EXPECT_EQ(result, (std::vector<uint32_t>{77, 2, 3, 11, 11}));
    std::vector<uint32_t> unchanged(3);
    source.memcopyTo(unchanged);
    EXPECT_EQ(unchanged, (std::vector<uint32_t>{1, 2, 3}));
}
