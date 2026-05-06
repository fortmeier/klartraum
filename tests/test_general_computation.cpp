#include <gtest/gtest.h>
#include <iostream>

#include "klartraum/headless_frontend.hpp"
#include "klartraum/vulkan_context.hpp"
#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/generalcomputation.hpp"
#include "klartraum/vulkan_buffer.hpp"

using namespace klartraum;

class GeneralComputationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize the headless frontend for testing
        frontend = std::make_unique<HeadlessFrontend>();
        engine = &frontend->getKlartraumEngine();
        vulkanContextPtr = &engine->getVulkanContext();
    }

    void TearDown() override {
        // Cleanup resources
        frontend.reset();
    }

    std::unique_ptr<HeadlessFrontend> frontend;
    KlartraumEngine* engine;
    VulkanContext* vulkanContextPtr;
};

TEST_F(GeneralComputationTest, ComputeGraphCreation) {
    VulkanContext& vulkanContext = *vulkanContextPtr;
    /*
    STEP 1: cerate the computegraph elements
    */

    typedef VulkanBuffer<float> typeA;
    typedef VulkanBuffer<float> typeB;
    typedef VulkanBuffer<float> typeR;
    typedef UniformBufferObject<float> typeU;

    std::string shaderPath = "shaders/operator_multiply_scalar_element_wise.comp.spv";

    auto op = std::make_shared<GeneralComputation<>>(vulkanContext, shaderPath);

    auto bufferElementA = std::make_shared<BufferElement<typeA>>(vulkanContext, 7);
    auto bufferElementB = std::make_shared<BufferElement<typeB>>(vulkanContext, 7);
    auto bufferElementR = std::make_shared<BufferElement<typeR>>(vulkanContext, 7);


    op->setInput(bufferElementA, 0);
    op->setInput(bufferElementB, 1);
    op->setInput(bufferElementR, 2);

    op->setGroupCountX(7);

    /*
    STEP 2: create the computegraph backend and compile the computegraph
    */
   
   // this traverses the computegraph and creates the vulkan objects
   auto computegraph = ComputeGraph(vulkanContext, 1);
   computegraph.enableProfiling();   // enable before compileFrom
   computegraph.compileFrom(op);

    std::vector<float> dataA = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    std::vector<float> dataB = {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f};

    bufferElementA->getBuffer(0).memcopyFrom(dataA);
    bufferElementB->getBuffer(0).memcopyFrom(dataB);

    /*
    STEP 3: submit the computegraph and compare the output
    */
    computegraph.submitAndWait(vulkanContext.getGraphicsQueue(), 0);

    // check the output buffer
    std::vector<float> data_out(7, 0.0f);

    op->getOutputElement<BufferElement<typeA>>(2)->getBuffer(0).memcopyTo(data_out);

    for (int i = 0; i < 7; i++) {
        EXPECT_EQ(dataA[i] * dataB[i], data_out[i]);
    }

    // Retrieve and verify profiling results.
    // submitAndWait() already accumulated the timestamps internally.
    auto profilingResults = computegraph.getProfilingResults();

    std::cout << "\n--- ComputeGraph profiling (single frame) ---\n";
    for (auto& [name, ms] : profilingResults) {
        std::cout << "  " << name << ": " << ms << " ms\n";
    }

    // There should be one entry per node in the graph
    // (the three leaf buffers + the op shader node).
    EXPECT_EQ(profilingResults.size(), 4u);

    // The compute shader node ('GeneralComputation') must have a positive time.
    // It's the last entry in topological order (root).
    EXPECT_GT(profilingResults.back().second, 0.0f)
        << "Compute shader reported zero GPU time — profiling may not be working";

    SUCCEED();
}