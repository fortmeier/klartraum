#include <gtest/gtest.h>

#include <map>
#include <vector>

#include "klartraum/glfw_frontend.hpp"

#include "klartraum/drawgraph/drawgraph.hpp"

#include "klartraum/drawgraph/imageviewsrc.hpp"
#include "klartraum/drawgraph/bufferelement.hpp"
#include "klartraum/drawgraph/buffertransformation.hpp"
#include "klartraum/drawgraph/uniformbufferobject.hpp"
#include "klartraum/vulkan_buffer.hpp"


using namespace klartraum;


TEST(BufferTransformation, create) {
    klartraum::GlfwFrontend frontend;

    auto& core = frontend.getKlartraumCore();
    auto& vulkanKernel = core.getVulkanKernel();
    auto& device = vulkanKernel.getDevice();

    typedef VulkanBuffer<float> typeA;
    typedef VulkanBuffer<float> typeR;
    auto transform = std::make_shared<BufferTransformation<typeA, typeR>>(vulkanKernel, "shaders/operator_double.comp.spv");
    
    auto bufferElement = std::make_shared<BufferElement<typeA>>(vulkanKernel, 7);
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    bufferElement->getBuffer().memcopy_from(data);
    
    transform->setInput(bufferElement);

    /*
    STEP 2: create the drawgraph backend and compile the drawgraph
    */
    
    // this traverses the drawgraph and creates the vulkan objects
    auto& drawgraph = DrawGraph(vulkanKernel, 1);
    drawgraph.compileFrom(transform);

    /*
    STEP 3: submit the drawgraph and compare the output
    */
    drawgraph.submitAndWait(vulkanKernel.getGraphicsQueue(), 0);

    // check the output buffer
    std::vector<float> data_out(7, 0.0f);

    transform->getOutputBuffer(0).memcopy_to(data_out);

    for (int i = 0; i < 7; i++) {
        EXPECT_EQ(data[i] * 2, data_out[i]);
    }
    return;
}

TEST(BufferTransformation, create_with_ubo) {
    klartraum::GlfwFrontend frontend;

    auto& core = frontend.getKlartraumCore();
    auto& vulkanKernel = core.getVulkanKernel();
    auto& device = vulkanKernel.getDevice();

    typedef VulkanBuffer<float> typeA;
    typedef VulkanBuffer<float> typeR;
    typedef UniformBufferObjectNew<float> typeU;

    auto transform = std::make_shared<BufferTransformation<typeA, typeR, typeU>>(vulkanKernel, "shaders/operator_multiply_scalar.comp.spv");
    
    auto bufferElement = std::make_shared<BufferElement<typeA>>(vulkanKernel, 7);
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    bufferElement->getBuffer().memcopy_from(data);
    
    transform->setInput(bufferElement);

    transform->getUbo()->ubo = 3.0f;

    /*
    STEP 2: create the drawgraph backend and compile the drawgraph
    */
    
    // this traverses the drawgraph and creates the vulkan objects
    auto& drawgraph = DrawGraph(vulkanKernel, 1);
    drawgraph.compileFrom(transform);

    /*
    STEP 3: submit the drawgraph and compare the output
    */
    drawgraph.submitAndWait(vulkanKernel.getGraphicsQueue(), 0);

    // check the output buffer
    std::vector<float> data_out(7, 0.0f);

    transform->getOutputBuffer(0).memcopy_to(data_out);

    for (int i = 0; i < 7; i++) {
        EXPECT_EQ(data[i] * 3.0f, data_out[i]);
    }
    return;
}
