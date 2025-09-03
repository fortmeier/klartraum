#include <vector>

#include <gtest/gtest.h>

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/generalcomputation.hpp"
#include "klartraum/computegraph/tensorelement.hpp"
#include "klartraum/glfw_frontend.hpp"
#include "klartraum/onnx/onnx_network.hpp"
#include "klartraum/onnx/onnx_push_constants.hpp"

using namespace klartraum;

class OnnxConvTransposeTest : public ::testing::Test {
protected:
    void SetUp() override {
        frontend = std::make_unique<GlfwFrontend>();
        vulkanContext = &frontend->getKlartraumEngine().getVulkanContext();
    }

    std::unique_ptr<GlfwFrontend> frontend;
    VulkanContext* vulkanContext;
};

TEST_F(OnnxConvTransposeTest, CreateConvTransposeTensors) {
    // Test creating tensors for ConvTranspose operation

    // Input tensor [1, 3, 2, 2] - batch=1, channels=3, height=2, width=2
    auto inputTensor = vulkanContext->create<TensorElement<float>>(
        std::vector<uint32_t>{1, 3, 2, 2});

    // Weight tensor [3, 16, 3, 3] - in_channels=3, out_channels=16, kernel_h=3, kernel_w=3
    auto weightTensor = vulkanContext->create<TensorElementSinglePath<float>>(
        std::vector<uint32_t>{3, 16, 3, 3});

    // Bias tensor [16] - one bias per output channel
    auto biasTensor = vulkanContext->create<TensorElementSinglePath<float>>(
        std::vector<uint32_t>{16});

    // Output tensor [1, 16, 4, 4] - upsampled output
    auto outputTensor = vulkanContext->create<TensorElement<float>>(
        std::vector<uint32_t>{1, 16, 4, 4});

    EXPECT_EQ(inputTensor->getDimensions()[0], 1);
    EXPECT_EQ(inputTensor->getDimensions()[1], 3);
    EXPECT_EQ(inputTensor->getDimensions()[2], 2);
    EXPECT_EQ(inputTensor->getDimensions()[3], 2);

    EXPECT_EQ(weightTensor->getDimensions()[0], 3);
    EXPECT_EQ(weightTensor->getDimensions()[1], 16);
    EXPECT_EQ(weightTensor->getDimensions()[2], 3);
    EXPECT_EQ(weightTensor->getDimensions()[3], 3);

    EXPECT_EQ(biasTensor->getDimensions()[0], 16);

    EXPECT_EQ(outputTensor->getDimensions()[0], 1);
    EXPECT_EQ(outputTensor->getDimensions()[1], 16);
    EXPECT_EQ(outputTensor->getDimensions()[2], 4);
    EXPECT_EQ(outputTensor->getDimensions()[3], 4);
}

TEST_F(OnnxConvTransposeTest, SetConvTransposeData) {
    // Test setting data in ConvTranspose tensors

    auto inputTensor = vulkanContext->create<TensorElement<float>>(
        std::vector<uint32_t>{1, 1, 2, 2} // Simple 2x2 single channel
    );

    auto weightTensor = vulkanContext->create<TensorElementSinglePath<float>>(
        std::vector<uint32_t>{1, 1, 3, 3} // Simple 3x3 kernel
    );

    auto biasTensor = vulkanContext->create<TensorElementSinglePath<float>>(
        std::vector<uint32_t>{1} // Single bias
    );

    // Initialize test data
    std::vector<float> inputData = {
        1.0f, 2.0f,
        3.0f, 4.0f
    };

    std::vector<float> weightData = {
        1.0f, 0.5f, 0.0f,
        0.5f, 1.0f, 0.5f,
        0.0f, 0.5f, 1.0f
    };

    std::vector<float> biasData = {0.1f};

    inputTensor->_setup(*vulkanContext, 1);
    weightTensor->_setup(*vulkanContext, 1);
    biasTensor->_setup(*vulkanContext, 1);

    // Set data in tensors
    inputTensor->setData(0, inputData);
    weightTensor->setData(0, weightData);
    biasTensor->setData(0, biasData);

    // Verify tensor data was set (dimensions should be preserved)
    EXPECT_EQ(inputTensor->getDataElementCount(), 4);
    EXPECT_EQ(weightTensor->getDataElementCount(), 9);
    EXPECT_EQ(biasTensor->getDataElementCount(), 1);
}

TEST_F(OnnxConvTransposeTest, ConvTransposePushConstants) {
    // Test that ConvTranspose push constants can be created

    ConvTransposePushConstants pushConstants = {};

    // Set typical conv transpose parameters
    pushConstants.dilations[0] = 1;
    pushConstants.dilations[1] = 1;
    pushConstants.groups[0] = 1;
    pushConstants.kernel_shape[0] = 3;
    pushConstants.kernel_shape[1] = 3;
    pushConstants.pads[0] = 1;
    pushConstants.pads[1] = 1;
    pushConstants.strides[0] = 2;
    pushConstants.strides[1] = 2;
    pushConstants.output_padding[0] = 1;
    pushConstants.output_padding[1] = 1;

    // Set tensor dimensions
    pushConstants.dimInput[0] = 1;
    pushConstants.dimInput[1] = 3;
    pushConstants.dimInput[2] = 2;
    pushConstants.dimInput[3] = 2;

    pushConstants.dimWeights[0] = 3;  // input channels
    pushConstants.dimWeights[1] = 16; // output channels
    pushConstants.dimWeights[2] = 3;
    pushConstants.dimWeights[3] = 3;

    pushConstants.dimBias[0] = 16;

    // Calculate expected output dimensions
    // output_size = (input_size - 1) * stride - 2 * padding + kernel_size + output_padding
    uint32_t expected_height = (2 - 1) * 2 - 2 * 1 + 3 + 1; // = 4
    uint32_t expected_width = (2 - 1) * 2 - 2 * 1 + 3 + 1;  // = 4

    pushConstants.dimOutput[0] = 1;
    pushConstants.dimOutput[1] = 16;
    pushConstants.dimOutput[2] = expected_height;
    pushConstants.dimOutput[3] = expected_width;

    EXPECT_EQ(pushConstants.kernel_shape[0], 3);
    EXPECT_EQ(pushConstants.kernel_shape[1], 3);
    EXPECT_EQ(pushConstants.strides[0], 2);
    EXPECT_EQ(pushConstants.strides[1], 2);
    EXPECT_EQ(pushConstants.dimInput[1], 3);    // input channels
    EXPECT_EQ(pushConstants.dimWeights[1], 16); // output channels
    EXPECT_EQ(pushConstants.dimOutput[2], 4);   // output height
    EXPECT_EQ(pushConstants.dimOutput[3], 4);   // output width
}

TEST_F(OnnxConvTransposeTest, ConvTransposeGeneralComputationFullTest) {
    // Test creating a complete GeneralComputation with conv transpose shader, tensors, and push constants

    // Create tensors for ConvTranspose operation
    // Input tensor [1, 2, 2, 2] - smaller for testing
    auto inputTensor = vulkanContext->create<TensorElement<float>>(
        std::vector<uint32_t>{1, 2, 2, 2});

    // Weight tensor [2, 3, 3, 3] - 2 input channels, 3 output channels, 3x3 kernel
    auto weightTensor = vulkanContext->create<TensorElementSinglePath<float>>(
        std::vector<uint32_t>{2, 3, 3, 3});

    // Bias tensor [3] - one bias per output channel
    auto biasTensor = vulkanContext->create<TensorElementSinglePath<float>>(
        std::vector<uint32_t>{3});

    // Output tensor [1, 3, 4, 4] - upsampled with stride=2
    auto outputTensor = vulkanContext->create<TensorElement<float>>(
        std::vector<uint32_t>{1, 3, 4, 4});

    // Create GeneralComputation with conv transpose shader
    std::string convTransposeShaderPath = "shaders/onnx/conv_transpose.comp.spv";
    auto convTransposeComputation = vulkanContext->create<GeneralComputation<ConvTransposePushConstants>>(
        convTransposeShaderPath);

    // Set up push constants
    ConvTransposePushConstants pushConstants = {};

    // ConvTranspose operation parameters
    pushConstants.dilations[0] = 1;
    pushConstants.dilations[1] = 1;
    pushConstants.groups[0] = 1;
    pushConstants.kernel_shape[0] = 3;
    pushConstants.kernel_shape[1] = 3;
    pushConstants.pads[0] = 1;
    pushConstants.pads[1] = 1;
    pushConstants.strides[0] = 2;
    pushConstants.strides[1] = 2;
    pushConstants.output_padding[0] = 1;
    pushConstants.output_padding[1] = 1;

    // Tensor dimensions
    pushConstants.dimInput[0] = 1; // batch
    pushConstants.dimInput[1] = 2; // input channels
    pushConstants.dimInput[2] = 2; // height
    pushConstants.dimInput[3] = 2; // width

    pushConstants.dimWeights[0] = 2; // input channels
    pushConstants.dimWeights[1] = 3; // output channels
    pushConstants.dimWeights[2] = 3; // kernel height
    pushConstants.dimWeights[3] = 3; // kernel width

    pushConstants.dimOutput[0] = 1; // batch
    pushConstants.dimOutput[1] = 3; // output channels
    pushConstants.dimOutput[2] = 4; // output height
    pushConstants.dimOutput[3] = 4; // output width

    pushConstants.dimBias[0] = 3; // output channels

    // Set push constants on the computation
    convTransposeComputation->setPushConstants({pushConstants});

    // Connect tensors as inputs/outputs to the computation
    convTransposeComputation->setInput(inputTensor, 0);  // input tensor
    convTransposeComputation->setInput(weightTensor, 1); // weight tensor
    convTransposeComputation->setInput(biasTensor, 2);   // bias tensor
    convTransposeComputation->setInput(outputTensor, 3); // output tensor (as buffer for writing)

    // Set compute dispatch dimensions
    // For conv transpose, dispatch based on output dimensions
    uint32_t outputWidth = 4;
    uint32_t outputHeight = 4;
    uint32_t outputChannels = 3;

    // Set group count to cover output dimensions (8x8 local size in shader)
    uint32_t groupsX = (outputWidth + 7) / 8; // ceil division
    uint32_t groupsY = (outputHeight + 7) / 8;
    uint32_t groupsZ = outputChannels;

    convTransposeComputation->setGroupCount(groupsX, groupsY, groupsZ);

    // Verify the computation was created successfully
    EXPECT_EQ(convTransposeComputation->getGroupCountX(), groupsX);
    EXPECT_EQ(convTransposeComputation->getGroupCountY(), groupsY);
    EXPECT_EQ(convTransposeComputation->getGroupCountZ(), groupsZ);

    auto computegraph = ComputeGraph(*vulkanContext, 1);
    computegraph.compileFrom(convTransposeComputation);

    // Create test data
    std::vector<float> inputData(1 * 2 * 2 * 2);
    std::vector<float> weightData(2 * 3 * 3 * 3);
    std::vector<float> biasData(3);

    // Fill with test values
    for (size_t i = 0; i < inputData.size(); ++i) {
        inputData[i] = static_cast<float>(i + 1); // 1, 2, 3, 4, 5, 6, 7, 8
    }

    for (size_t i = 0; i < weightData.size(); ++i) {
        weightData[i] = 0.1f * static_cast<float>(i + 1); // 0.1, 0.2, 0.3, ...
    }

    for (size_t i = 0; i < biasData.size(); ++i) {
        biasData[i] = 0.5f + static_cast<float>(i); // 0.5, 1.5, 2.5
    }

    // Set data in tensors
    inputTensor->setData(0, inputData);
    weightTensor->setData(0, weightData);
    biasTensor->setData(0, biasData);

    // Verify tensor data was set correctly
    EXPECT_EQ(inputTensor->getDataElementCount(), 8);   // 1*2*2*2
    EXPECT_EQ(weightTensor->getDataElementCount(), 54); // 2*3*3*3
    EXPECT_EQ(biasTensor->getDataElementCount(), 3);
    EXPECT_EQ(outputTensor->getDataElementCount(), 48); // 1*3*4*4

    // Verify push constants were set correctly
    EXPECT_EQ(pushConstants.kernel_shape[0], 3);
    EXPECT_EQ(pushConstants.kernel_shape[1], 3);
    EXPECT_EQ(pushConstants.strides[0], 2);
    EXPECT_EQ(pushConstants.strides[1], 2);
    EXPECT_EQ(pushConstants.dimInput[1], 2);    // input channels
    EXPECT_EQ(pushConstants.dimWeights[1], 3);  // output channels
    EXPECT_EQ(pushConstants.dimBias[0], 3);

    std::cout << "ConvTransposeGeneralComputationFullTest: Created complete ConvTranspose computation with:" << std::endl;
    std::cout << "  - Input tensor: [1, 2, 2, 2] = " << inputTensor->getDataElementCount() << " elements" << std::endl;
    std::cout << "  - Weight tensor: [2, 3, 3, 3] = " << weightTensor->getDataElementCount() << " elements" << std::endl;
    std::cout << "  - Bias tensor: [3] = " << biasTensor->getDataElementCount() << " elements" << std::endl;
    std::cout << "  - Output tensor: [1, 3, 4, 4] = " << outputTensor->getDataElementCount() << " elements" << std::endl;
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
}
