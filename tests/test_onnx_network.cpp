#include <filesystem>
#include <fstream>

#include <gtest/gtest.h>

#include "klartraum/headless_frontend.hpp"
#include "klartraum/onnx/onnx_network.hpp"
#include "klartraum/computegraph/tensorelement.hpp"
#include "onnx.pb.h"

using namespace klartraum;


void testLayer(std::shared_ptr<OnnxNetwork> onnxNetwork, std::string layerName)
{
    auto element = onnxNetwork->getOutputElement(layerName);
    ASSERT_NE(element, nullptr);
    auto tensor = std::dynamic_pointer_cast<TensorElement<float>>(element);
    ASSERT_NE(tensor, nullptr);

    std::vector<float> dataGPU( tensor->getDataElementCount() );
    tensor->getDataBuffer(0).memcopyTo(dataGPU);

    // now compare the output tensor to expected values
    std::vector<float> dataGT = onnxNetwork->getFloatInitializerData(layerName);

    ASSERT_EQ(dataGT.size(), dataGPU.size());

    for(size_t i = 0; i < dataGT.size(); i++) {
        ASSERT_NEAR(dataGT[i], dataGPU[i], 1e-3);
    }
}

// Test execute functionality
TEST(OnnxNetworkTest, ExecuteWithValidEncoderModel) {
    HeadlessFrontend frontend;
    auto& core = frontend.getKlartraumEngine();
    auto& vulkanContext = core.getVulkanContext();

    /*
    STEP 1: create the ONNX network
    */
    std::string modelPath = "./data/onnx/simple_encoder_with_onnx_frozen_intermediates.onnx";

    auto onnxNetwork = vulkanContext.create<OnnxNetwork>(modelPath);


    /*
    STEP 2: use the render engine to execute the computegraph so it can be debugged with renderdoc
    */
    core.add(core.createRenderPass());
    core.add(onnxNetwork);
    core.step();

    testLayer(onnxNetwork, "/conv1/Conv_output_0");
    testLayer(onnxNetwork, "/relu/Relu_output_0");
    testLayer(onnxNetwork, "/conv2/Conv_output_0");
    testLayer(onnxNetwork, "/relu_1/Relu_output_0");
    testLayer(onnxNetwork, "/conv3/Conv_output_0");
    testLayer(onnxNetwork, "output");

    return;
}

TEST(OnnxNetworkTest, ExecuteWithValidDecoderModel) {
    HeadlessFrontend frontend;
    auto& core = frontend.getKlartraumEngine();
    auto& vulkanContext = core.getVulkanContext();

    /*
    STEP 1: create the ONNX network
    */
    std::string modelPath = "./data/onnx/simple_decoder_with_onnx_frozen_intermediates.onnx";

    auto onnxNetwork = vulkanContext.create<OnnxNetwork>(modelPath);

    /*
    STEP 2: use the render engine to execute the computegraph so it can be debugged with renderdoc
    */
    core.add(core.createRenderPass());
    core.add(onnxNetwork);
    core.step();

    testLayer(onnxNetwork, "/deconv1/ConvTranspose_output_0");
    testLayer(onnxNetwork, "/relu/Relu_output_0");
    testLayer(onnxNetwork, "/deconv2/ConvTranspose_output_0");
    testLayer(onnxNetwork, "/relu_1/Relu_output_0");
    testLayer(onnxNetwork, "/deconv3/ConvTranspose_output_0");
    testLayer(onnxNetwork, "output");

    return;
}

