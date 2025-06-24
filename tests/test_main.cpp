#include <gtest/gtest.h>

#include "klartraum/glfw_frontend.hpp"
#include "klartraum/vulkan_buffer.hpp"


TEST(KlartraumGlfwFrontend, smoke) {
    klartraum::GlfwFrontend frontend;

    auto& core = frontend.getKlartraumEngine();
    auto& vulkanContext = core.getVulkanContext();
    auto& device = vulkanContext.getDevice();
    
}