#include <vector>

#include <gtest/gtest.h>

#include "klartraum/computegraph/generalcomputation.hpp"
#include "klartraum/computegraph/tensorelement.hpp"
#include "klartraum/glfw_frontend.hpp"
#include "klartraum/onnx_network.hpp"
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

TEST_F(OnnxTransposeTest, Transpose2D) {
    auto inputTensor = vulkanContext->create<TensorElement<float>>(std::vector<uint32_t>{2, 3});
    auto outputTensor = vulkanContext->create<TensorElement<float>>(std::vector<uint32_t>{3, 2});
    TransposePushConstants pushConstants = {};
    pushConstants.dims = 2;
    pushConstants.dimPerm[0] = 1;
    pushConstants.dimPerm[1] = 0;
    pushConstants.dimInput[0] = 2;
    pushConstants.dimInput[1] = 3;
    pushConstants.dimOutput[0] = 3;
    pushConstants.dimOutput[1] = 2;
    std::string shaderPath = "shaders/onnx/transpose2.comp.spv";
    auto transposeComputation = vulkanContext->create<GeneralComputation<TransposePushConstants>>(shaderPath);
    transposeComputation->setPushConstants({pushConstants});
    transposeComputation->setInput(inputTensor, 0);
    transposeComputation->setInput(outputTensor, 1);
    transposeComputation->setGroupCount(2, 3, 1);
    ComputeGraph computeGraph(*vulkanContext, 1);
    computeGraph.compileFrom(transposeComputation);
    std::vector<float> inputData = {1, 2, 3,
                                    4, 5, 6};
    inputTensor->setData(0, inputData);
    computeGraph.submitAndWait(vulkanContext->getGraphicsQueue(), 0);
    std::vector<float> expected = {1, 4,
                                   2, 5,
                                   3, 6};
    std::vector<float> readBack(6);
    outputTensor->getDataBuffer(0).memcopyTo(readBack);
    EXPECT_EQ(readBack, expected);
}
    TransposePushConstants pushConstants = {};

TEST_F(OnnxTransposeTest, Transpose4D) {
    auto inputTensor = vulkanContext->create<TensorElement<float>>(std::vector<uint32_t>{1, 2, 4, 4});
    auto outputTensor = vulkanContext->create<TensorElement<float>>(std::vector<uint32_t>{1, 4, 4, 2});
    TransposePushConstants pushConstants = {};
    pushConstants.dims = 4;
    pushConstants.dimPerm[0] = 0;
    pushConstants.dimPerm[1] = 2;
    pushConstants.dimPerm[2] = 3;
    pushConstants.dimPerm[3] = 1;
    pushConstants.dimInput[0] = 1;
    pushConstants.dimInput[1] = 2;
    pushConstants.dimInput[2] = 4;
    pushConstants.dimInput[3] = 4;
    pushConstants.dimOutput[0] = 1;
    pushConstants.dimOutput[1] = 4;
    pushConstants.dimOutput[2] = 4;
    pushConstants.dimOutput[3] = 2;
    std::string shaderPath = "shaders/onnx/transpose4.comp.spv";
    auto transposeComputation = vulkanContext->create<GeneralComputation<TransposePushConstants>>(shaderPath);
    transposeComputation->setPushConstants({pushConstants});
    transposeComputation->setInput(inputTensor, 0);
    transposeComputation->setInput(outputTensor, 1);
    transposeComputation->setGroupCount(2, 4, 4);
    ComputeGraph computeGraph(*vulkanContext, 1);
    computeGraph.compileFrom(transposeComputation);
    std::vector<float> inputData = {
        1.0f, 1.0, 1.0f, 1.0, 1.0f, 1.0, 1.0f, 1.0, 1.0f, 1.0, 1.0f, 1.0, 1.0f, 1.0, 1.0f, 1.0,
        2.0f, 2.0, 2.0f, 2.0, 2.0f, 2.0, 2.0f, 2.0, 2.0f, 2.0, 2.0f, 2.0, 2.0f, 2.0, 2.0f, 2.0,
    };
    inputTensor->setData(0, inputData);
    computeGraph.submitAndWait(vulkanContext->getGraphicsQueue(), 0);
    std::vector<float> outputData = {
        1.0f, 2.0, 1.0f, 2.0, 1.0f, 2.0, 1.0f, 2.0, 1.0f, 2.0, 1.0f, 2.0, 1.0f, 2.0, 1.0f, 2.0,
        1.0f, 2.0, 1.0f, 2.0, 1.0f, 2.0, 1.0f, 2.0, 1.0f, 2.0, 1.0f, 2.0, 1.0f, 2.0, 1.0f, 2.0,
    };
    EXPECT_EQ(inputTensor->getDataElementCount(), outputTensor->getDataElementCount());
    std::vector<float> readBackData(32);
    outputTensor->getDataBuffer(0).memcopyTo(readBackData);
    EXPECT_EQ(readBackData, outputData);
}
