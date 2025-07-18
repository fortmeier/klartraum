#include <gtest/gtest.h>

#include <map>
#include <vector>

#include "klartraum/glfw_frontend.hpp"

#include "klartraum/computegraph/computegraph.hpp"

#include "klartraum/computegraph/imageviewsrc.hpp"
#include "klartraum/computegraph/bufferelement.hpp"
#include "klartraum/computegraph/buffertransformation.hpp"
#include "klartraum/computegraph/uniformbufferobject.hpp"
#include "klartraum/vulkan_buffer.hpp"


using namespace klartraum;


TEST(BufferTransformation, create) {
    klartraum::GlfwFrontend frontend;

    auto& core = frontend.getKlartraumEngine();
    auto& vulkanContext = core.getVulkanContext();
    auto& device = vulkanContext.getDevice();

    typedef VulkanBuffer<float> typeA;
    typedef VulkanBuffer<float> typeR;
    auto transform = std::make_shared<BufferTransformation<typeA, typeR>>(vulkanContext, "shaders/operator_double.comp.spv");
    
    auto bufferElement = std::make_shared<BufferElement<typeA>>(vulkanContext, 7);
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    
    transform->setInput(bufferElement);
    
    /*
    STEP 2: create the computegraph backend and compile the computegraph
    */
   
    // this traverses the computegraph and creates the vulkan objects
    auto computegraph = ComputeGraph(vulkanContext, 1);
    computegraph.compileFrom(transform);

    bufferElement->getBuffer(0).memcopyFrom(data);

    /*
    STEP 3: submit the computegraph and compare the output
    */
    computegraph.submitAndWait(vulkanContext.getGraphicsQueue(), 0);

    // check the output buffer
    std::vector<float> data_out(7, 0.0f);

    transform->getOutputBuffer(0).memcopyTo(data_out);

    for (int i = 0; i < 7; i++) {
        EXPECT_EQ(data[i] * 2, data_out[i]);
    }
    return;
}

TEST(BufferTransformation, create_with_ubo) {
    klartraum::GlfwFrontend frontend;

    auto& core = frontend.getKlartraumEngine();
    auto& vulkanContext = core.getVulkanContext();
    auto& device = vulkanContext.getDevice();

    /*
    STEP 1: cerate the computegraph elements
    */

    typedef VulkanBuffer<float> typeA;
    typedef VulkanBuffer<float> typeR;
    typedef UniformBufferObject<float> typeU;

    auto transform = std::make_shared<BufferTransformation<typeA, typeR, typeU>>(vulkanContext, "shaders/operator_multiply_scalar.comp.spv");
    
    auto bufferElement = std::make_shared<BufferElement<typeA>>(vulkanContext, 7);
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    
    transform->setInput(bufferElement);
    
    transform->getUbo()->ubo = 3.0f;
    
    /*
    STEP 2: create the computegraph backend and compile the computegraph
    */
   
   // this traverses the computegraph and creates the vulkan objects
   auto computegraph = ComputeGraph(vulkanContext, 1);
   computegraph.compileFrom(transform);

   bufferElement->getBuffer(0).memcopyFrom(data);

    /*
    STEP 3: submit the computegraph and compare the output
    */
    computegraph.submitAndWait(vulkanContext.getGraphicsQueue(), 0);

    // check the output buffer
    std::vector<float> data_out(7, 0.0f);

    transform->getOutputBuffer(0).memcopyTo(data_out);

    for (int i = 0; i < 7; i++) {
        EXPECT_EQ(data[i] * 3.0f, data_out[i]);
    }
    return;
}

TEST(BufferTransformation, create_with_ubo_multiple_paths) {
    klartraum::GlfwFrontend frontend;

    auto& core = frontend.getKlartraumEngine();
    auto& vulkanContext = core.getVulkanContext();
    auto& device = vulkanContext.getDevice();

    typedef VulkanBuffer<float> typeA;
    typedef VulkanBuffer<float> typeR;
    typedef UniformBufferObject<float> typeU;

    auto transform = std::make_shared<BufferTransformation<typeA, typeR, typeU>>(vulkanContext, "shaders/operator_multiply_scalar.comp.spv");
    
    auto bufferElement = std::make_shared<BufferElement<typeA>>(vulkanContext, 7);
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    
    transform->setInput(bufferElement);
    
    transform->getUbo()->ubo = 77.0f;
    
    /*
    STEP 2: create the computegraph backend and compile the computegraph and copy the data to the buffer
    */
   
   // this traverses the computegraph and creates the vulkan objects
   auto computegraph = ComputeGraph(vulkanContext, 3);
   computegraph.compileFrom(transform);
   
   bufferElement->getBuffer(0).memcopyFrom(data);
   /*
   STEP 3: submit the computegraph and compare the output
   */
  for(uint32_t pathId = 0; pathId < 3; pathId++) {
        transform->getUbo()->ubo = 1.0f * pathId;
        transform->getUbo()->update(pathId);

        computegraph.submitAndWait(vulkanContext.getGraphicsQueue(), pathId);
        
        // check the output buffer
        std::vector<float> data_out(7, 0.0f);
        
        transform->getOutputBuffer(pathId).memcopyTo(data_out);
        
        for (int i = 0; i < 7; i++) {
            EXPECT_EQ(data[i] * 1.0f * pathId, data_out[i]);
        }
    }
    return;
}
