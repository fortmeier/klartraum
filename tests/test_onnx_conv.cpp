#include <gtest/gtest.h>
#include <vector>

#include "klartraum/glfw_frontend.hpp"
#include "klartraum/onnx_network.hpp"
#include "klartraum/computegraph/tensorelement.hpp"
#include "klartraum/computegraph/generalcomputation.hpp"
#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/onnx_push_constants.hpp"

using namespace klartraum;

class OnnxConvTest : public ::testing::Test {
protected:
    void SetUp() override {
        frontend = std::make_unique<GlfwFrontend>();
        vulkanContext = &frontend->getKlartraumEngine().getVulkanContext();
    }

    std::unique_ptr<GlfwFrontend> frontend;
    VulkanContext* vulkanContext;
};

TEST_F(OnnxConvTest, CreateConvTensors) {
    // Test creating tensors for Conv operation
    
    // Input tensor [1, 3, 8, 8] - batch=1, channels=3, height=8, width=8
    auto inputTensor = vulkanContext->create<TensorElement<float>>(
        std::vector<uint32_t>{1, 3, 8, 8}
    );
    
    // Weight tensor [16, 3, 3, 3] - out_channels=16, in_channels=3, kernel_h=3, kernel_w=3
    auto weightTensor = vulkanContext->create<TensorElementSinglePath<float>>(
        std::vector<uint32_t>{16, 3, 3, 3}
    );
    
    // Bias tensor [16] - one bias per output channel
    auto biasTensor = vulkanContext->create<TensorElementSinglePath<float>>(
        std::vector<uint32_t>{16}
    );
    
    // Output tensor [1, 16, 6, 6] - assuming no padding, stride=1
    auto outputTensor = vulkanContext->create<TensorElement<float>>(
        std::vector<uint32_t>{1, 16, 6, 6}
    );
    
    EXPECT_EQ(inputTensor->getDimensions()[0], 1);
    EXPECT_EQ(inputTensor->getDimensions()[1], 3);
    EXPECT_EQ(inputTensor->getDimensions()[2], 8);
    EXPECT_EQ(inputTensor->getDimensions()[3], 8);
    
    EXPECT_EQ(weightTensor->getDimensions()[0], 16);
    EXPECT_EQ(weightTensor->getDimensions()[1], 3);
    EXPECT_EQ(weightTensor->getDimensions()[2], 3);
    EXPECT_EQ(weightTensor->getDimensions()[3], 3);
    
    EXPECT_EQ(biasTensor->getDimensions()[0], 16);
    
    EXPECT_EQ(outputTensor->getDimensions()[0], 1);
    EXPECT_EQ(outputTensor->getDimensions()[1], 16);
    EXPECT_EQ(outputTensor->getDimensions()[2], 6);
    EXPECT_EQ(outputTensor->getDimensions()[3], 6);
}

TEST_F(OnnxConvTest, SetConvData) {
    // Test setting data in Conv tensors
    
    auto inputTensor = vulkanContext->create<TensorElement<float>>(
        std::vector<uint32_t>{1, 1, 4, 4}  // Simple 4x4 single channel
    );
    
    auto weightTensor = vulkanContext->create<TensorElementSinglePath<float>>(
        std::vector<uint32_t>{1, 1, 3, 3}  // Simple 3x3 kernel
    );
    
    auto biasTensor = vulkanContext->create<TensorElementSinglePath<float>>(
        std::vector<uint32_t>{1}  // Single bias
    );
    
    // Initialize test data
    std::vector<float> inputData = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f
    };
    
    std::vector<float> weightData = {
        1.0f, 0.0f, -1.0f,
        1.0f, 0.0f, -1.0f,
        1.0f, 0.0f, -1.0f
    };
    
    std::vector<float> biasData = {0.5f};

    inputTensor->_setup(*vulkanContext, 1);
    weightTensor->_setup(*vulkanContext, 1);
    biasTensor->_setup(*vulkanContext, 1);

    // Set data in tensors
    inputTensor->setData(0, inputData);
    weightTensor->setData(0, weightData);
    biasTensor->setData(0, biasData);
    
    // Verify tensor data was set (dimensions should be preserved)
    EXPECT_EQ(inputTensor->getDataElementCount(), 16);
    EXPECT_EQ(weightTensor->getDataElementCount(), 9);
    EXPECT_EQ(biasTensor->getDataElementCount(), 1);
}

TEST_F(OnnxConvTest, ConvPushConstants) {
    // Test that Conv push constants can be created
    
    ConvPushConstants pushConstants = {};
    
    // Set typical conv parameters
    pushConstants.dilations[0] = 1;
    pushConstants.dilations[1] = 1;
    pushConstants.groups[0] = 1;
    pushConstants.kernel_shape[0] = 3;
    pushConstants.kernel_shape[1] = 3;
    pushConstants.pads[0] = 0;
    pushConstants.pads[1] = 0;
    pushConstants.strides[0] = 1;
    pushConstants.strides[1] = 1;
    
    // Set tensor dimensions
    pushConstants.dimInput[0] = 1;
    pushConstants.dimInput[1] = 3;
    pushConstants.dimInput[2] = 8;
    pushConstants.dimInput[3] = 8;
    
    pushConstants.dimWeights[0] = 16;
    pushConstants.dimWeights[1] = 3;
    pushConstants.dimWeights[2] = 3;
    pushConstants.dimWeights[3] = 3;
    
    pushConstants.dimBias[0] = 16;
    
    EXPECT_EQ(pushConstants.kernel_shape[0], 3);
    EXPECT_EQ(pushConstants.kernel_shape[1], 3);
    EXPECT_EQ(pushConstants.strides[0], 1);
    EXPECT_EQ(pushConstants.strides[1], 1);
    EXPECT_EQ(pushConstants.dimInput[1], 3);  // input channels
    EXPECT_EQ(pushConstants.dimWeights[0], 16);  // output channels
}

TEST_F(OnnxConvTest, ConvGeneralComputationFullTest) {
    // Test creating a complete GeneralComputation with conv shader, tensors, and push constants
    
    // Create tensors for Conv operation
    // Input tensor [1, 2, 4, 4] - smaller for testing
    auto inputTensor = vulkanContext->create<TensorElement<float>>(
        std::vector<uint32_t>{1, 2, 4, 4}
    );
    
    // Weight tensor [3, 2, 3, 3] - 3 output channels, 2 input channels, 3x3 kernel
    auto weightTensor = vulkanContext->create<TensorElementSinglePath<float>>(
        std::vector<uint32_t>{3, 2, 3, 3}
    );
    
    // Bias tensor [3] - one bias per output channel
    auto biasTensor = vulkanContext->create<TensorElementSinglePath<float>>(
        std::vector<uint32_t>{3}
    );
    
    // Output tensor [1, 3, 2, 2] - with 3x3 kernel on 4x4 input, output is 2x2 (no padding, stride=1)
    auto outputTensor = vulkanContext->create<TensorElement<float>>(
        std::vector<uint32_t>{1, 3, 2, 2}
    );
    
    // Create GeneralComputation with conv shader
    std::string convShaderPath = "shaders/onnx/conv.comp.spv";
    auto convComputation = vulkanContext->create<GeneralComputation<ConvPushConstants>>(
        convShaderPath
    );
    
    // Set up push constants
    ConvPushConstants pushConstants = {};
    
    // Conv operation parameters
    pushConstants.dilations[0] = 1;
    pushConstants.dilations[1] = 1;
    pushConstants.groups[0] = 1;
    pushConstants.kernel_shape[0] = 3;
    pushConstants.kernel_shape[1] = 3;
    pushConstants.pads[0] = 0;  // no padding
    pushConstants.pads[1] = 0;
    pushConstants.strides[0] = 1;
    pushConstants.strides[1] = 1;
    
    // Tensor dimensions
    pushConstants.dimInput[0] = 1;  // batch
    pushConstants.dimInput[1] = 2;  // input channels
    pushConstants.dimInput[2] = 4;  // height
    pushConstants.dimInput[3] = 4;  // width
    
    pushConstants.dimWeights[0] = 3;  // output channels
    pushConstants.dimWeights[1] = 2;  // input channels
    pushConstants.dimWeights[2] = 3;  // kernel height
    pushConstants.dimWeights[3] = 3;  // kernel width
    
    pushConstants.dimBias[0] = 3;  // output channels
    
    // Set push constants on the computation
    convComputation->setPushConstants({pushConstants});
    
    // Connect tensors as inputs/outputs to the computation
    convComputation->setInput(inputTensor, 0);    // input tensor
    convComputation->setInput(weightTensor, 1);   // weight tensor
    convComputation->setInput(biasTensor, 2);     // bias tensor
    convComputation->setInput(outputTensor, 3);   // output tensor (as buffer for writing)
    
    // Set compute dispatch dimensions
    // For conv, typically dispatch based on output dimensions
    uint32_t outputWidth = 2;
    uint32_t outputHeight = 2;
    uint32_t outputChannels = 3;
    
    // Set group count to cover output dimensions (8x8 local size in shader)
    uint32_t groupsX = (outputWidth + 7) / 8;   // ceil division
    uint32_t groupsY = (outputHeight + 7) / 8;
    uint32_t groupsZ = outputChannels;
    
    convComputation->setGroupCount(groupsX, groupsY, groupsZ);
    
    // // Setup tensors
    // inputTensor->_setup(*vulkanContext, 1);
    // weightTensor->_setup(*vulkanContext, 1);
    // biasTensor->_setup(*vulkanContext, 1);
    // outputTensor->_setup(*vulkanContext, 1);

    // Setup the computation with VulkanContext
    //convComputation->_setup(*vulkanContext, 1);  // 1 path
    
    // Verify the computation was created successfully
    EXPECT_EQ(convComputation->getGroupCountX(), groupsX);
    EXPECT_EQ(convComputation->getGroupCountY(), groupsY);
    EXPECT_EQ(convComputation->getGroupCountZ(), groupsZ);
    
    auto computegraph = ComputeGraph(*vulkanContext, 1);
    computegraph.compileFrom(convComputation);
    
    // Create test data
    std::vector<float> inputData(1 * 2 * 4 * 4);
    std::vector<float> weightData(3 * 2 * 3 * 3);
    std::vector<float> biasData(3);
    
    // Fill with test values
    for (size_t i = 0; i < inputData.size(); ++i) {
        inputData[i] = static_cast<float>(i + 1);  // 1, 2, 3, ...
    }
    
    for (size_t i = 0; i < weightData.size(); ++i) {
        weightData[i] = 0.1f * static_cast<float>(i + 1);  // 0.1, 0.2, 0.3, ...
    }
    
    for (size_t i = 0; i < biasData.size(); ++i) {
        biasData[i] = 0.5f + static_cast<float>(i);  // 0.5, 1.5, 2.5
    }
    
    // Set data in tensors
    inputTensor->setData(0, inputData);
    weightTensor->setData(0, weightData);
    biasTensor->setData(0, biasData);
    
    // Verify tensor data was set correctly
    EXPECT_EQ(inputTensor->getDataElementCount(), 32);   // 1*2*4*4
    EXPECT_EQ(weightTensor->getDataElementCount(), 54);  // 3*2*3*3
    EXPECT_EQ(biasTensor->getDataElementCount(), 3);
    EXPECT_EQ(outputTensor->getDataElementCount(), 12);  // 1*3*2*2
    
    // Verify push constants were set correctly
    EXPECT_EQ(pushConstants.kernel_shape[0], 3);
    EXPECT_EQ(pushConstants.kernel_shape[1], 3);
    EXPECT_EQ(pushConstants.dimInput[1], 2);
    EXPECT_EQ(pushConstants.dimWeights[0], 3);
    EXPECT_EQ(pushConstants.dimBias[0], 3);
    
    std::cout << "ConvGeneralComputationFullTest: Created complete Conv computation with:" << std::endl;
    std::cout << "  - Input tensor: [1, 2, 4, 4] = " << inputTensor->getDataElementCount() << " elements" << std::endl;
    std::cout << "  - Weight tensor: [3, 2, 3, 3] = " << weightTensor->getDataElementCount() << " elements" << std::endl;
    std::cout << "  - Bias tensor: [3] = " << biasTensor->getDataElementCount() << " elements" << std::endl;
    std::cout << "  - Output tensor: [1, 3, 2, 2] = " << outputTensor->getDataElementCount() << " elements" << std::endl;
    std::cout << "  - Compute groups: [" << groupsX << ", " << groupsY << ", " << groupsZ << "]" << std::endl;



    computegraph.submitAndWait(vulkanContext->getGraphicsQueue(), 0);

    // Copy output tensor data back to main memory
    std::vector<float> outputData(outputTensor->getDataElementCount());
    outputTensor->getDataBuffer(0).memcopyTo(outputData);

    // Print the output for verification
    std::cout << "Output tensor data:" << std::endl;
    for (size_t i = 0; i < outputData.size(); ++i) {
        std::cout << "  [" << i << "] = " << outputData[i] << std::endl;
    }

    return;
}

