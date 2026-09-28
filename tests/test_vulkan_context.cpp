/**
 * TESTS:
 * - deviceSelection: VulkanContext selects a non-CPU (hardware) Vulkan device
 * - deviceFeatures: VulkanContext reports the features available on the selected device
 * - minimalComputePipeline: a minimal SPIR-V compute pipeline can be created and destroyed without error
 * - minimalComputeDispatch: dispatching the minimal pipeline produces correct output
 **/

#include <gtest/gtest.h>
#include <iostream>
#include <vector>

#include <vulkan/vulkan.h>

#include "klartraum/headless_frontend.hpp"
#include "klartraum/vulkan_context.hpp"
#include "klartraum/vulkan_helpers.hpp"
#include "klartraum/vulkan_buffer.hpp"

using namespace klartraum;

class VulkanContextTest : public ::testing::Test {
protected:
    void SetUp() override {
        frontend = std::make_unique<HeadlessFrontend>();
        vc = &frontend->getKlartraumEngine().getVulkanContext();
    }
    void TearDown() override { frontend.reset(); }

    std::unique_ptr<HeadlessFrontend> frontend;
    VulkanContext* vc = nullptr;
};

// ----------------------------------------------------------------
// Test: deviceSelection
// Checks that the selected Vulkan physical device is not a CPU
// (i.e. not LavaPipe / software renderer).
// ----------------------------------------------------------------
TEST_F(VulkanContextTest, deviceSelection) {
    VkPhysicalDevice pd = vc->getPhysicalDevice();
    ASSERT_NE(pd, VK_NULL_HANDLE);

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(pd, &props);

    std::cout << "  deviceName:  " << props.deviceName << "\n";
    std::cout << "  deviceType:  " << props.deviceType << "\n";
    std::cout << "  apiVersion:  "
              << VK_VERSION_MAJOR(props.apiVersion) << "."
              << VK_VERSION_MINOR(props.apiVersion) << "."
              << VK_VERSION_PATCH(props.apiVersion) << "\n";
    std::cout << "  driverVersion: " << props.driverVersion << "\n";
    std::cout << "  vendorID:    0x" << std::hex << props.vendorID << std::dec << "\n";

    EXPECT_NE(props.deviceType, VK_PHYSICAL_DEVICE_TYPE_CPU)
        << "Selected a CPU/software renderer (" << props.deviceName
        << "). Release mode may be using LavaPipe.";
}

// ----------------------------------------------------------------
// Test: deviceFeatures
// Prints all features the physical device supports and the subset
// actually requested at device creation. Useful to compare what
// Debug (with validation layer overrides) vs Release enables.
// ----------------------------------------------------------------
TEST_F(VulkanContextTest, deviceFeatures) {
    VkPhysicalDevice pd = vc->getPhysicalDevice();

    VkPhysicalDeviceFeatures supported{};
    vkGetPhysicalDeviceFeatures(pd, &supported);

    std::cout << "  --- Physical device feature support ---\n";
    std::cout << "  robustBufferAccess:                    " << supported.robustBufferAccess << "\n";
    std::cout << "  shaderInt64:                           " << supported.shaderInt64 << "\n";
    std::cout << "  shaderInt16:                           " << supported.shaderInt16 << "\n";
    std::cout << "  shaderFloat64:                         " << supported.shaderFloat64 << "\n";
    std::cout << "  fragmentStoresAndAtomics:              " << supported.fragmentStoresAndAtomics << "\n";
    std::cout << "  vertexPipelineStoresAndAtomics:        " << supported.vertexPipelineStoresAndAtomics << "\n";
    std::cout << "  pipelineStatisticsQuery:               " << supported.pipelineStatisticsQuery << "\n";
    std::cout << "  shaderStorageBufferArrayDynamicIndexing: " << supported.shaderStorageBufferArrayDynamicIndexing << "\n";

    // Check VK_KHR_shader_non_semantic_info (needed if shaders use GL_EXT_debug_printf)
    uint32_t cnt = 0;
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &cnt, nullptr);
    std::vector<VkExtensionProperties> exts(cnt);
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &cnt, exts.data());
    bool hasNonSemantic = false;
    bool hasScalarLayout = false;
    for (auto& e : exts) {
        if (strcmp(e.extensionName, "VK_KHR_shader_non_semantic_info") == 0) hasNonSemantic = true;
        if (strcmp(e.extensionName, "VK_EXT_scalar_block_layout")       == 0) hasScalarLayout = true;
    }
    std::cout << "  VK_KHR_shader_non_semantic_info:       " << hasNonSemantic << "\n";
    std::cout << "  VK_EXT_scalar_block_layout:            " << hasScalarLayout << "\n";

    SUCCEED();
}

// ----------------------------------------------------------------
// Test: minimalComputePipeline
// Creates the same pipeline that BufferTransformation.create uses
// (operator_double.comp) directly via raw Vulkan calls, without
// going through the BufferTransformation wrapper. If this crashes
// in Release but not Debug, the problem is in pipeline creation
// itself (e.g. missing device feature or extension).
// ----------------------------------------------------------------
TEST_F(VulkanContextTest, minimalComputePipeline) {
    VkDevice device = vc->getDevice();

    // Descriptor set layout: binding 0 (input), binding 2 (output)
    VkDescriptorSetLayoutBinding bindings[2] = {};
    bindings[0].binding         = 0;
    bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].binding         = 2;
    bindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutCI{};
    layoutCI.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutCI.bindingCount = 2;
    layoutCI.pBindings    = bindings;

    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    ASSERT_EQ(vkCreateDescriptorSetLayout(device, &layoutCI, nullptr, &setLayout), VK_SUCCESS);
    std::cout << "  DescriptorSetLayout created: " << setLayout << "\n";

    // Pipeline layout
    VkPipelineLayoutCreateInfo pipelineLayoutCI{};
    pipelineLayoutCI.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutCI.setLayoutCount = 1;
    pipelineLayoutCI.pSetLayouts    = &setLayout;

    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    ASSERT_EQ(vkCreatePipelineLayout(device, &pipelineLayoutCI, nullptr, &pipelineLayout), VK_SUCCESS);
    std::cout << "  PipelineLayout created:       " << pipelineLayout << "\n";

    // Load the same shader
    auto code = readFile("shaders/operator_double.comp.spv");
    std::cout << "  Shader SPIR-V size: " << code.size() << " bytes\n";
    VkShaderModule shaderModule = createShaderModule(code, device);
    std::cout << "  ShaderModule created:         " << shaderModule << "\n";

    // Compute pipeline
    VkPipelineShaderStageCreateInfo stageCI{};
    stageCI.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageCI.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    stageCI.module = shaderModule;
    stageCI.pName  = "main";

    VkComputePipelineCreateInfo pipelineCI{};
    pipelineCI.sType              = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineCI.stage              = stageCI;
    pipelineCI.layout             = pipelineLayout;
    pipelineCI.basePipelineHandle = VK_NULL_HANDLE;
    pipelineCI.basePipelineIndex  = -1;

    std::cout << "  Calling vkCreateComputePipelines...\n";
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkResult result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineCI, nullptr, &pipeline);
    std::cout << "  vkCreateComputePipelines returned: " << result << "\n";
    ASSERT_EQ(result, VK_SUCCESS);
    std::cout << "  Pipeline created:             " << pipeline << "\n";

    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyShaderModule(device, shaderModule, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, setLayout, nullptr);
    std::cout << "  All objects destroyed cleanly.\n";
}

// ----------------------------------------------------------------
// Test: minimalComputeDispatch
// Actually dispatches operator_double on 7 floats and checks output.
// Isolates whether the crash is in pipeline creation or dispatch.
// ----------------------------------------------------------------
TEST_F(VulkanContextTest, minimalComputeDispatch) {
    VkDevice device = vc->getDevice();

    const uint32_t N = 7;
    std::vector<float> inputData  = {1, 2, 3, 4, 5, 6, 7};
    std::vector<float> outputData(N, 0.0f);

    VulkanBuffer<float> inputBuf (*vc, N);
    VulkanBuffer<float> outputBuf(*vc, N);
    inputBuf.memcopyFrom(inputData);

    // Descriptor set layout: binding 0 = input, binding 2 = output
    VkDescriptorSetLayoutBinding bindings[2] = {};
    bindings[0].binding = 0; bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1; bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].binding = 2; bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1; bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutCI{};
    layoutCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutCI.bindingCount = 2; layoutCI.pBindings = bindings;
    VkDescriptorSetLayout setLayout{};
    ASSERT_EQ(vkCreateDescriptorSetLayout(device, &layoutCI, nullptr, &setLayout), VK_SUCCESS);

    VkPipelineLayoutCreateInfo plCI{};
    plCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plCI.setLayoutCount = 1; plCI.pSetLayouts = &setLayout;
    VkPipelineLayout pipelineLayout{};
    ASSERT_EQ(vkCreatePipelineLayout(device, &plCI, nullptr, &pipelineLayout), VK_SUCCESS);

    auto code = readFile("shaders/operator_double.comp.spv");
    VkShaderModule shaderModule = createShaderModule(code, device);

    VkPipelineShaderStageCreateInfo stageCI{};
    stageCI.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageCI.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageCI.module = shaderModule; stageCI.pName = "main";

    VkComputePipelineCreateInfo pipelineCI{};
    pipelineCI.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineCI.stage = stageCI; pipelineCI.layout = pipelineLayout;
    pipelineCI.basePipelineHandle = VK_NULL_HANDLE; pipelineCI.basePipelineIndex = -1;

    VkPipeline pipeline{};
    ASSERT_EQ(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineCI, nullptr, &pipeline), VK_SUCCESS);
    vkDestroyShaderModule(device, shaderModule, nullptr);

    // Descriptor pool + set
    VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
    VkDescriptorPoolCreateInfo poolCI{};
    poolCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolCI.maxSets = 1; poolCI.poolSizeCount = 1; poolCI.pPoolSizes = &poolSize;
    VkDescriptorPool pool{};
    ASSERT_EQ(vkCreateDescriptorPool(device, &poolCI, nullptr, &pool), VK_SUCCESS);

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = pool; allocInfo.descriptorSetCount = 1; allocInfo.pSetLayouts = &setLayout;
    VkDescriptorSet descSet{};
    ASSERT_EQ(vkAllocateDescriptorSets(device, &allocInfo, &descSet), VK_SUCCESS);

    VkDescriptorBufferInfo inInfo{inputBuf.getBuffer(), 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo outInfo{outputBuf.getBuffer(), 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet writes[2] = {};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[0].dstSet = descSet;
    writes[0].dstBinding = 0; writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[0].pBufferInfo = &inInfo;
    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[1].dstSet = descSet;
    writes[1].dstBinding = 2; writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[1].pBufferInfo = &outInfo;
    vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);

    // Record and submit
    VkCommandBufferAllocateInfo cmdAI{};
    cmdAI.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAI.commandPool = vc->getCommandPool();
    cmdAI.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cmdAI.commandBufferCount = 1;
    VkCommandBuffer cmd{};
    ASSERT_EQ(vkAllocateCommandBuffers(device, &cmdAI, &cmd), VK_SUCCESS);

    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descSet, 0, nullptr);
    vkCmdDispatch(cmd, N, 1, 1);
    vkEndCommandBuffer(cmd);

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    vkQueueSubmit(vc->getGraphicsQueue(), 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(vc->getGraphicsQueue());
    vkFreeCommandBuffers(device, vc->getCommandPool(), 1, &cmd);

    outputBuf.memcopyTo(outputData);

    for (uint32_t i = 0; i < N; i++) {
        EXPECT_FLOAT_EQ(outputData[i], inputData[i] * 2.0f)
            << "index " << i << ": got " << outputData[i] << " expected " << inputData[i] * 2.0f;
    }

    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyDescriptorPool(device, pool, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, setLayout, nullptr);
}
