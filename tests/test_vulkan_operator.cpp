#include <gtest/gtest.h>

#include "klartraum/glfw_frontend.hpp"
#include "klartraum/vulkan_buffer.hpp"
#include "klartraum/vulkan_operator.hpp"


TEST(VulkanOperator, simple_matmul) {
    klartraum::GlfwFrontend frontend;

    
    auto& core = frontend.getKlartraumCore();
    auto& vulkanKernel = core.getVulkanKernel();
    auto& device = vulkanKernel.getDevice();

    typedef glm::mat4 Mat4;
    typedef klartraum::VulkanBuffer<Mat4> BufferMat;

    typedef glm::vec4 Vec4;
    typedef klartraum::VulkanBuffer<Vec4> BufferVec;
    
    BufferMat bufferMat(vulkanKernel, 2);
    BufferVec bufferVec(vulkanKernel, 2);
    

    std::vector<Mat4> dataMat = {Mat4(1.0f), Mat4(2.0f)};
    bufferMat.memcopy_from(dataMat);

    std::vector<Vec4> dataVec = {Vec4(1.0f, 0.0f, 0.0f, 1.0f), Vec4(1.0f, 0.0f, 0.0f, 1.0f)};
    bufferVec.memcopy_from(dataVec);
    
    BufferVec result(vulkanKernel, 2);

    std::vector<Vec4> result_data(2);
    
    auto matmul = klartraum::VulkanOperator<BufferMat, BufferVec, BufferVec>(vulkanKernel, "shaders/operator_matmul_M4V4.comp.spv");
    auto commandBuffer = vulkanKernel.createCommandBuffer();
    klartraum::VulkanOperationContext context(commandBuffer, device); // = vulkanKernel.createOperationContext();
    context(matmul(bufferMat, bufferVec, result));
    context.eval(vulkanKernel.getGraphicsQueue());
    
    result.memcopy_to(result_data);

    EXPECT_EQ(result_data[0][0], 1.0f);
    EXPECT_EQ(result_data[0][1], 0.0f);
    EXPECT_EQ(result_data[0][2], 0.0f);
    EXPECT_EQ(result_data[0][3], 1.0f);

    EXPECT_EQ(result_data[1][0], 2.0f);
    EXPECT_EQ(result_data[1][1], 0.0f);
    EXPECT_EQ(result_data[1][2], 0.0f);
    EXPECT_EQ(result_data[1][3], 2.0f);

    return;
}