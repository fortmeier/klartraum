// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - memcopy: data copied into a VulkanBuffer and back to the host is unchanged
 **/
#include <gtest/gtest.h>

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