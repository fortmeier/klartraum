// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "klartraum/headless_frontend.hpp"
#include "klartraum/vulkan_buffer.hpp"


TEST(KlartraumHeadlessFrontend, smoke) {
    klartraum::HeadlessFrontend frontend;

    auto& core = frontend.getKlartraumEngine();
    auto& vulkanContext = core.getVulkanContext();
    auto& device = vulkanContext.getDevice();

}
