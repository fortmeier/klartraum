#include <gtest/gtest.h>
#include <vector>

#include "klartraum/glfw_frontend.hpp"
#include "klartraum/onnx_network.hpp"
#include "klartraum/computegraph/tensorelement.hpp"
#include "klartraum/computegraph/generalcomputation.hpp"
#include "klartraum/onnx_push_constants.hpp"

using namespace klartraum;

class OnnxTransposeTest : public ::testing::Test {
protected:
    void SetUp() override {
        frontend = std::make_unique<GlfwFrontend>();
        vulkanContext = &frontend->getKlartraumEngine().getVulkanContext();
    }

    std::unique_ptr<GlfwFrontend> frontend;
    VulkanContext* vulkanContext;
};

TEST_F(OnnxTransposeTest, CreateTransposeTensors) {
    // Test creating tensors and running a transpose operation using GeneralComputation

    // Input tensor [1, 2, 4, 4] - NCHW format
    auto inputTensor = vulkanContext->create<TensorElement<float>>(
        std::vector<uint32_t>{1, 2, 4, 4}
    );

    // Output tensor [1, 4, 4, 2] - NHWC format (transposed)
    auto outputTensor = vulkanContext->create<TensorElement<float>>(
        std::vector<uint32_t>{1, 4, 4, 2}
    );

    // Set up TransposePushConstants
    TransposePushConstants pushConstants = {};
    pushConstants.dims = 4; // 4D tensor

    // Typical NCHW to NHWC permutation: [0, 2, 3, 1]
    pushConstants.dimPerm[0] = 0;
    pushConstants.dimPerm[1] = 2;
    pushConstants.dimPerm[2] = 3;
    pushConstants.dimPerm[3] = 1;

    // Set input/output dimensions in push constants
    pushConstants.dimInput[0] = 1;
    pushConstants.dimInput[1] = 2;
    pushConstants.dimInput[2] = 4;
    pushConstants.dimInput[3] = 4;

    pushConstants.dimOutput[0] = 1;
    pushConstants.dimOutput[1] = 4;
    pushConstants.dimOutput[2] = 4;
    pushConstants.dimOutput[3] = 2;

    // Create GeneralComputation node with transpose shader
    std::string transposeShaderPath = "shaders/onnx/transpose4.comp.spv";
    auto transposeComputation = vulkanContext->create<GeneralComputation<TransposePushConstants>>(
        transposeShaderPath
    );
    transposeComputation->setPushConstants({pushConstants});

    // Connect tensors as inputs/outputs
    transposeComputation->setInput(inputTensor, 0);    // input tensor
    transposeComputation->setInput(outputTensor, 1);   // output tensor

    // Set group count (for demonstration, use output dimensions)
    uint32_t groupsX = 2;
    uint32_t groupsY = 4;
    uint32_t groupsZ = 4;
    transposeComputation->setGroupCount(groupsX, groupsY, groupsZ);

    // Create and compile compute graph
    ComputeGraph computeGraph(*vulkanContext, 1);
    computeGraph.compileFrom(transposeComputation);

    // create input data
    std::vector<float> inputData = {
        // NCHW format: [batch, channel, height, width]
        // batch = 1, channel = 2, height = 4, width = 4
        // total number of elements = 32 = 1 * 2 * 4 * 4
        // before transpose, all elements of c=0 are 1.0
        // and all elements of c=1 are 2.0
        1.0f, 1.0, 1.0f, 1.0, 1.0f, 1.0, 1.0f, 1.0, 1.0f, 1.0, 1.0f, 1.0, 1.0f, 1.0, 1.0f, 1.0,
        2.0f, 2.0, 2.0f, 2.0, 2.0f, 2.0, 2.0f, 2.0, 2.0f, 2.0, 2.0f, 2.0, 2.0f, 2.0, 2.0f, 2.0,
    };

    inputTensor->setData(0, inputData);

    // Execute the computation
    computeGraph.submitAndWait(vulkanContext->getGraphicsQueue(), 0);

    // create expected output data
    std::vector<float> outputData = {
        // NCHW format: [batch, height, width, channel]
        // after transpose, all elements of c=0 are 1.0
        // and all elements of c=1 are 2.0
        // but these are now interleaved
        1.0f, 2.0, 1.0f, 2.0, 1.0f, 2.0, 1.0f, 2.0, 1.0f, 2.0, 1.0f, 2.0, 1.0f, 2.0, 1.0f, 2.0,
        1.0f, 2.0, 1.0f, 2.0, 1.0f, 2.0, 1.0f, 2.0, 1.0f, 2.0, 1.0f, 2.0, 1.0f, 2.0, 1.0f, 2.0,
    };


    // Total elements should be the same
    EXPECT_EQ(inputTensor->getDataElementCount(), outputTensor->getDataElementCount());

    // Read back output data
    std::vector<float> readBackData(32);
    outputTensor->getDataBuffer(0).memcopyTo(readBackData);
    EXPECT_EQ(readBackData, outputData);
}
