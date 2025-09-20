#include <filesystem>
#include <fstream>

#include <gtest/gtest.h>

#include "klartraum/glfw_frontend.hpp"
#include "klartraum/onnx/onnx_network.hpp"
#include "klartraum/computegraph/tensorelement.hpp"
#include "onnx.pb.h"

using namespace klartraum;


void testLayer(std::shared_ptr<OnnxNetwork> onnxNetwork, std::string layerName)
{
    auto conv1Output = onnxNetwork->getOutputElement(layerName);
    ASSERT_NE(conv1Output, nullptr);
    auto conv1OutputTensor = std::dynamic_pointer_cast<TensorElement<float>>(conv1Output);
    ASSERT_NE(conv1OutputTensor, nullptr);

    std::vector<float> dataGPU( conv1OutputTensor->getDataElementCount() );
    conv1OutputTensor->getDataBuffer(0).memcopyTo(dataGPU);

    // now compare the output tensor to expected values
    std::vector<float> dataGT = onnxNetwork->getFloatInitializerData(layerName);

    for(size_t i = 0; i < dataGT.size(); i++) {
        ASSERT_NEAR(dataGT[i], dataGPU[i], 1e-3);
    }
}

// Test execute functionality
TEST(OnnxNetworkTest, ExecuteWithValidModel) {
    GlfwFrontend frontend;
    auto& core = frontend.getKlartraumEngine();
    auto& vulkanContext = core.getVulkanContext();

    /*
    STEP 1: create the ONNX network
    */
    std::string modelPath = "./data/onnx/simple_encoder_with_onnx_frozen_intermediates.onnx";

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
    testLayer(onnxNetwork, "/conv1/Conv_output_0");
    testLayer(onnxNetwork, "/relu/Relu_output_0");

    return;
}

