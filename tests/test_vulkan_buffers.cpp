#include <gtest/gtest.h>

#include "klartraum/glfw_frontend.hpp"
#include "klartraum/vulkan_buffer.hpp"


TEST(VulkanBuffer, memcopy) {
    klartraum::GlfwFrontend frontend;

    auto& core = frontend.getKlartraumCore();
    auto& vulkanKernel = core.getVulkanKernel();
    auto& device = vulkanKernel.getDevice();
    
    klartraum::VulkanBuffer<float> buffer(vulkanKernel, 7);
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    buffer.memcopy_from(data);
    std::vector<float> data2(7);
    buffer.memcopy_to(data2);
    for (int i = 0; i < 7; i++) {
        EXPECT_EQ(data[i], data2[i]);
    }
    return;
}