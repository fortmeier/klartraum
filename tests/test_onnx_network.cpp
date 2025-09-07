#include <filesystem>
#include <fstream>

#include <gtest/gtest.h>

#include "klartraum/glfw_frontend.hpp"
#include "klartraum/onnx/onnx_network.hpp"
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
    std::string modelPath = "./data/onnx/simple_encoder_with_value_info_and_param_info.onnx";

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
// Test execute functionality with decoder model
TEST(OnnxNetworkTest, CheckIncompleteModelLoadFails) {
    /**
     * For now, the ONNX loader only supports loading of models that have
     * value infos for all tensors in the model.
     * If this will be changed in the future and tensor shape etc. can be inferred
     * automatically, this test will need to be updated and might be deleted.
     */
    GlfwFrontend frontend;
    auto& core = frontend.getKlartraumEngine();
    auto& vulkanContext = core.getVulkanContext();

    /*
    STEP 1: create the ONNX network
    */
    std::string modelPath = "./data/onnx/simple_decoder.onnx";

    try {
        auto onnxNetwork = vulkanContext.create<OnnxNetwork>(modelPath);
        FAIL() << "Expected runtime_error to be thrown";
    } catch (const std::runtime_error& e) {
        // Expected exception caught
        SUCCEED();
    }

    return;
}

