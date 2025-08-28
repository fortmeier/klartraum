#include <vector>

#include <gtest/gtest.h>

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/generalcomputation.hpp"
#include "klartraum/computegraph/tensorelement.hpp"
#include "klartraum/glfw_frontend.hpp"
#include "klartraum/onnx_network.hpp"
#include "klartraum/onnx_push_constants.hpp"

using namespace klartraum;

class OnnxReluTest : public ::testing::Test {
protected:
    void SetUp() override {
        frontend = std::make_unique<GlfwFrontend>();
        vulkanContext = &frontend->getKlartraumEngine().getVulkanContext();
    }

    std::unique_ptr<GlfwFrontend> frontend;
    VulkanContext* vulkanContext;
};

TEST_F(OnnxReluTest, ReluGeneralComputationFullTest) {
    // Test creating a complete GeneralComputation with ReLU shader, tensors, and push constants

    // Create input and output tensors for ReLU operation
    // Input tensor [1, 2, 3, 4] - batch=1, channels=2, height=3, width=4
    auto inputTensor = vulkanContext->create<TensorElement<float>>(
        std::vector<uint32_t>{1, 2, 3, 4});

    // Output tensor [1, 2, 3, 4] - ReLU keeps same dimensions
    auto outputTensor = vulkanContext->create<TensorElement<float>>(
        std::vector<uint32_t>{1, 2, 3, 4});

    // Create GeneralComputation with ReLU shader
    std::string reluShaderPath = "shaders/onnx/relu.comp.spv";
    auto reluComputation = vulkanContext->create<GeneralComputation<TensorOpPushConstants>>(
        reluShaderPath);

    // Set up push constants for ReLU
    TensorOpPushConstants pushConstants = {};

    // Input and output dimensions are the same for ReLU
    pushConstants.dimInput[0] = 1; // batch
    pushConstants.dimInput[1] = 2; // channels
    pushConstants.dimInput[2] = 3; // height
    pushConstants.dimInput[3] = 4; // width

    pushConstants.dimOutput[0] = 1; // batch
    pushConstants.dimOutput[1] = 2; // channels
    pushConstants.dimOutput[2] = 3; // height
    pushConstants.dimOutput[3] = 4; // width

    // Set push constants on the computation
    reluComputation->setPushConstants({pushConstants});

    // Connect tensors as inputs/outputs to the computation
    reluComputation->setInput(inputTensor, 0);  // input tensor
    reluComputation->setInput(outputTensor, 1); // output tensor (as buffer for writing)

    // Set compute dispatch dimensions
    // For ReLU, dispatch based on tensor dimensions
    uint32_t totalElements = 1 * 2 * 3 * 4;                                 // 24 elements total
    uint32_t workgroupSize = 64;                                            // Typical workgroup size
    uint32_t groupsX = (totalElements + workgroupSize - 1) / workgroupSize; // ceil division
    uint32_t groupsY = 1;
    uint32_t groupsZ = 1;

    reluComputation->setGroupCount(groupsX, groupsY, groupsZ);

    // Verify the computation was created successfully
    EXPECT_EQ(reluComputation->getGroupCountX(), groupsX);
    EXPECT_EQ(reluComputation->getGroupCountY(), groupsY);
    EXPECT_EQ(reluComputation->getGroupCountZ(), groupsZ);

    auto computegraph = ComputeGraph(*vulkanContext, 1);
    computegraph.compileFrom(reluComputation);

    // Create test data with positive and negative values
    std::vector<float> inputData = {
        // First channel (2x3x4 = 24 elements total, split into 2 channels of 12 each)
        -2.5f, -1.0f, 0.0f, 1.5f, // row 1
        -0.5f, 2.0f, -3.0f, 0.5f, // row 2
        4.0f, -1.5f, 3.5f, -0.1f, // row 3
                                  // Second channel
        1.0f, -2.0f, 0.0f, -0.5f, // row 1
        3.0f, 0.5f, -1.0f, 2.5f,  // row 2
        -4.0f, 1.5f, -0.1f, 5.0f  // row 3
    };

    // Expected ReLU output (max(0, x))
    std::vector<float> expectedOutput = {
        // First channel - ReLU applied
        0.0f, 0.0f, 0.0f, 1.5f, // row 1
        0.0f, 2.0f, 0.0f, 0.5f, // row 2
        4.0f, 0.0f, 3.5f, 0.0f, // row 3
                                // Second channel - ReLU applied
        1.0f, 0.0f, 0.0f, 0.0f, // row 1
        3.0f, 0.5f, 0.0f, 2.5f, // row 2
        0.0f, 1.5f, 0.0f, 5.0f  // row 3
    };

    // Set data in input tensor
    inputTensor->setData(0, inputData);

    // Verify tensor data was set correctly
    EXPECT_EQ(inputTensor->getDataElementCount(), 24);  // 1*2*3*4
    EXPECT_EQ(outputTensor->getDataElementCount(), 24); // 1*2*3*4

    // Verify push constants were set correctly
    EXPECT_EQ(pushConstants.dimInput[0], 1);
    EXPECT_EQ(pushConstants.dimInput[1], 2);
    EXPECT_EQ(pushConstants.dimInput[2], 3);
    EXPECT_EQ(pushConstants.dimInput[3], 4);
    EXPECT_EQ(pushConstants.dimOutput[0], 1);
    EXPECT_EQ(pushConstants.dimOutput[1], 2);
    EXPECT_EQ(pushConstants.dimOutput[2], 3);
    EXPECT_EQ(pushConstants.dimOutput[3], 4);

    // Execute the computation
    computegraph.submitAndWait(vulkanContext->getGraphicsQueue(), 0);

    // Copy output tensor data back to main memory
    std::vector<float> outputData(outputTensor->getDataElementCount());
    outputTensor->getDataBuffer(0).memcopyTo(outputData);

    // Verify the ReLU operation worked correctly
    const float epsilon = 1e-6f;
    for (size_t i = 0; i < expectedOutput.size(); ++i) {
        EXPECT_NEAR(outputData[i], expectedOutput[i], epsilon)
            << "ReLU output mismatch at index " << i
            << ": expected " << expectedOutput[i]
            << ", got " << outputData[i]
            << " (input was " << inputData[i] << ")";
    }

    // Additional verification: ensure all negative values became zero
    for (size_t i = 0; i < inputData.size(); ++i) {
        if (inputData[i] <= 0.0f) {
            EXPECT_NEAR(outputData[i], 0.0f, epsilon)
                << "Negative/zero input at index " << i
                << " (value: " << inputData[i] << ") should produce zero output, got " << outputData[i];
        } else {
            EXPECT_NEAR(outputData[i], inputData[i], epsilon)
                << "Positive input at index " << i
                << " (value: " << inputData[i] << ") should be unchanged, got " << outputData[i];
        }
    }
}