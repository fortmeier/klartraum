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

    /*
    STEP 1: create the ONNX network
    */
    std::string modelPath = "C:\\Users\\dfort\\Desktop\\workspace\\super_resolution\\super-resolution-10.onnx";

    auto onnxNetwork = vulkanContext.create<OnnxNetwork>(modelPath);

    /*
    STEP 2: create the computegraph backend and compile the computegraph
    */

    // this traverses the computegraph and creates the vulkan objects
    auto computegraph = ComputeGraph(vulkanContext, 1);
    computegraph.compileFrom(onnxNetwork);

    /*
    STEP 3: submit the computegraph and compare the output
    */
    computegraph.submitAndWait(vulkanContext.getGraphicsQueue(), 0);

    return;
}
