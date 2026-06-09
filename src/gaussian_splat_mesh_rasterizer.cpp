#include <stdexcept>
#include <array>

#include "klartraum/gaussian_splat_mesh_rasterizer.hpp"
#include "klartraum/vulkan_helpers.hpp"

namespace klartraum {

GaussianSplatMeshRasterizer::GaussianSplatMeshRasterizer(
    std::vector<std::shared_ptr<BufferElementInterface>> splatBuffers,
    std::shared_ptr<BufferElementInterface> meshArgsBuffer)
    : splatBuffers(std::move(splatBuffers)), meshArgsBuffer(std::move(meshArgsBuffer)) {
}

GaussianSplatMeshRasterizer::~GaussianSplatMeshRasterizer() {
    auto& device = vulkanContext->getDevice();
    vkDestroyPipeline(device, graphicsPipeline, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(device, splatDescriptorSetLayout, nullptr);
}

void GaussianSplatMeshRasterizer::initialize(VulkanContext& vulkanContext, VkRenderPass& renderPass, std::shared_ptr<CameraUboType> cameraUBO) {
    DrawComponent::initialize(vulkanContext, renderPass, cameraUBO);

    createSplatDescriptorSetLayout();
    createDescriptorPool();
    createDescriptorSets();
    createGraphicsPipeline();
}

void GaussianSplatMeshRasterizer::createSplatDescriptorSetLayout() {
    auto& device = vulkanContext->getDevice();

    std::vector<VkDescriptorSetLayoutBinding> layoutBindings(splatBuffers.size());
    for (size_t i = 0; i < splatBuffers.size(); i++) {
        layoutBindings[i].binding = (uint32_t)i;
        layoutBindings[i].descriptorCount = 1;
        layoutBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[i].pImmutableSamplers = nullptr;
        layoutBindings[i].stageFlags = VK_SHADER_STAGE_MESH_BIT_EXT;
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = (uint32_t)layoutBindings.size();
    layoutInfo.pBindings = layoutBindings.data();

    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &splatDescriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create mesh splat descriptor set layout!");
    }
}

void GaussianSplatMeshRasterizer::createDescriptorPool() {
    auto& device = vulkanContext->getDevice();
    uint32_t numberPaths = vulkanContext->getNumberOfSwapChainImages();

    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = numberPaths * (uint32_t)splatBuffers.size();

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    poolInfo.maxSets = numberPaths;

    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create mesh splat descriptor pool!");
    }
}

void GaussianSplatMeshRasterizer::createDescriptorSets() {
    auto& device = vulkanContext->getDevice();
    uint32_t numberPaths = vulkanContext->getNumberOfSwapChainImages();

    std::vector<VkDescriptorSetLayout> layouts(numberPaths, splatDescriptorSetLayout);

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = numberPaths;
    allocInfo.pSetLayouts = layouts.data();

    splatDescriptorSets.resize(numberPaths);
    if (vkAllocateDescriptorSets(device, &allocInfo, splatDescriptorSets.data()) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate mesh splat descriptor sets!");
    }

    for (uint32_t pathId = 0; pathId < numberPaths; pathId++) {
        std::vector<VkDescriptorBufferInfo> bufferInfos(splatBuffers.size());
        std::vector<VkWriteDescriptorSet> writes(splatBuffers.size());

        for (size_t i = 0; i < splatBuffers.size(); i++) {
            bufferInfos[i].buffer = splatBuffers[i]->getVkBuffer(pathId);
            bufferInfos[i].offset = 0;
            bufferInfos[i].range = splatBuffers[i]->getBufferMemSize();

            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = splatDescriptorSets[pathId];
            writes[i].dstBinding = (uint32_t)i;
            writes[i].dstArrayElement = 0;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].descriptorCount = 1;
            writes[i].pBufferInfo = &bufferInfos[i];
        }

        vkUpdateDescriptorSets(device, (uint32_t)writes.size(), writes.data(), 0, nullptr);
    }
}

void GaussianSplatMeshRasterizer::createGraphicsPipeline() {
    auto device = vulkanContext->getDevice();
    auto swapChainExtent = vulkanContext->getSwapChainExtent();

    auto meshShaderCode = readFile("shaders/gsplat/gsplat_raster.mesh.spv");
    auto fragShaderCode = readFile("shaders/gsplat/gsplat_raster.frag.spv");

    VkShaderModule meshShaderModule = createShaderModule(meshShaderCode, device);
    VkShaderModule fragShaderModule = createShaderModule(fragShaderCode, device);

    VkPipelineShaderStageCreateInfo meshShaderStageInfo{};
    meshShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    meshShaderStageInfo.stage = VK_SHADER_STAGE_MESH_BIT_EXT;
    meshShaderStageInfo.module = meshShaderModule;
    meshShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
    fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.module = fragShaderModule;
    fragShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo shaderStages[] = { meshShaderStageInfo, fragShaderStageInfo };

    std::vector<VkDynamicState> dynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();

    // Mesh pipelines have no vertex-input or input-assembly stage; the mesh
    // shader emits primitives directly (pVertexInputState/pInputAssemblyState
    // are ignored and left null).

    VkViewport viewport{};
    viewport.x = 0.0f; viewport.y = 0.0f;
    viewport.width = (float)swapChainExtent.width;
    viewport.height = (float)swapChainExtent.height;
    viewport.minDepth = 0.0f; viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.offset = { 0, 0 };
    scissor.extent = swapChainExtent;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisampling.minSampleShading = 1.0f;

    // Premultiplied "over" (matches the vertex path / gsplat_raster.frag output).
    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_TRUE;
    colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
    colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;

    VkDescriptorSetLayout setLayouts[] = { cameraUBO->getDescriptorSetLayout(), splatDescriptorSetLayout };

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 2;
    pipelineLayoutInfo.pSetLayouts = setLayouts;
    pipelineLayoutInfo.pushConstantRangeCount = 0;

    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create mesh pipeline layout!");
    }

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = nullptr;
    pipelineInfo.pInputAssemblyState = nullptr;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = nullptr;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.renderPass = *renderPass;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
    pipelineInfo.basePipelineIndex = -1;

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
        throw std::runtime_error("failed to create mesh graphics pipeline!");
    }

    vkDestroyShaderModule(device, fragShaderModule, nullptr);
    vkDestroyShaderModule(device, meshShaderModule, nullptr);
}

void GaussianSplatMeshRasterizer::recordCommandBuffer(VkCommandBuffer commandBuffer, VkFramebuffer framebuffer, uint32_t pathId) {
    auto& swapChainExtent = vulkanContext->getSwapChainExtent();

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

    VkViewport viewport{};
    viewport.x = 0.0f; viewport.y = 0.0f;
    viewport.width = static_cast<float>(swapChainExtent.width);
    viewport.height = static_cast<float>(swapChainExtent.height);
    viewport.minDepth = 0.0f; viewport.maxDepth = 1.0f;
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset = { 0, 0 };
    scissor.extent = swapChainExtent;
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

    auto& cameraDescriptorSets = cameraUBO->getDescriptorSets();
    VkDescriptorSet sets[] = { cameraDescriptorSets[pathId], splatDescriptorSets[pathId] };
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 2, sets, 0, nullptr);

    auto drawMeshTasks = vulkanContext->getCmdDrawMeshTasksIndirectEXT();
    drawMeshTasks(commandBuffer, meshArgsBuffer->getVkBuffer(pathId), 0, 1,
                  sizeof(VkDrawMeshTasksIndirectCommandEXT));
}

} // namespace klartraum
