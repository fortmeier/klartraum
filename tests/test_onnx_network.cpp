#include <filesystem>
#include <fstream>

#include <gtest/gtest.h>

#include "klartraum/glfw_frontend.hpp"
#include "klartraum/onnx_network.hpp"
#include "onnx.pb.h"

using namespace klartraum;

// Test execute functionality
TEST(OnnxNetworkTest, ExecuteWithValidModel) {
    GlfwFrontend frontend;
    auto& core = frontend.getKlartraumEngine();
    auto& vulkanContext = core.getVulkanContext();

    std::string modelPath = "C:\\Users\\dfort\\Desktop\\workspace\\super_resolution\\super-resolution-10.onnx";

    auto onnxNetwork = std::make_unique<OnnxNetwork>(vulkanContext, modelPath);
}
