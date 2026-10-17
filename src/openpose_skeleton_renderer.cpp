// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "klartraum/openpose_skeleton_renderer.hpp"
#include "klartraum/vulkan_helpers.hpp"

namespace klartraum {

namespace {

// The annotator draws at 512 pixels with stick width and joint radius 4.
float defaultStyleSize(VkExtent2D extent) {
    return 4.0f * static_cast<float>(std::min(extent.width, extent.height)) / 512.0f;
}

} // namespace

OpenPoseSkeletonRenderer::OpenPoseSkeletonRenderer(std::shared_ptr<BufferElementInterface> joints,
                                                   uint32_t skeletonCount, VkExtent2D targetExtent)
    : joints(std::move(joints)),
      skeletonCount(skeletonCount),
      targetExtent(targetExtent),
      stickWidth(defaultStyleSize(targetExtent)),
      jointRadius(defaultStyleSize(targetExtent)) {}

OpenPoseSkeletonRenderer::~OpenPoseSkeletonRenderer() {
    // Nothing was created if initialize() never ran.
    if (vulkanContext == nullptr) {
        return;
    }
    auto& device = vulkanContext->getDevice();
    vkDestroyPipeline(device, graphicsPipeline, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(device, jointDescriptorSetLayout, nullptr);
}

void OpenPoseSkeletonRenderer::setOcclusionDepth(std::shared_ptr<BufferElementInterface> depth, float relativeBias) {
    if (!depth) {
        throw std::runtime_error("OpenPoseSkeletonRenderer: the occlusion depth buffer is missing");
    }
    occlusionDepth = std::move(depth);
    occlusionBias = relativeBias;
}

void OpenPoseSkeletonRenderer::initialize(VulkanContext& vulkanContext, VkRenderPass& renderPass,
                                          std::shared_ptr<CameraUboType> cameraUBO) {
    DrawComponent::initialize(vulkanContext, renderPass, cameraUBO);
    if (joints->getBufferMemSize() < sizeof(glm::vec4) * openpose::kJointCount * skeletonCount) {
        throw std::runtime_error("OpenPoseSkeletonRenderer: joint buffer is smaller than its skeletons");
    }
    if (occlusionDepth &&
        occlusionDepth->getBufferMemSize() < sizeof(float) * targetExtent.width * targetExtent.height) {
        throw std::runtime_error("OpenPoseSkeletonRenderer: the occlusion depth buffer is smaller than the target");
    }
    createDescriptorSets();
    createGraphicsPipeline();
}

void OpenPoseSkeletonRenderer::createDescriptorSets() {
    auto& device = vulkanContext->getDevice();

    // Binding 0: joints, for the vertex shader; binding 1: occlusion depth, for the fragment shader (the joints again
    // when there is none; then it is not read).
    VkDescriptorSetLayoutBinding bindings[2]{};
    for (uint32_t i = 0; i < 2; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorCount = 1;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].stageFlags = i == 0 ? VK_SHADER_STAGE_VERTEX_BIT : VK_SHADER_STAGE_FRAGMENT_BIT;
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 2;
    layoutInfo.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &jointDescriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create joint descriptor set layout!");
    }

    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 2 * numberPaths;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    poolInfo.maxSets = numberPaths;
    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create joint descriptor pool!");
    }

    std::vector<VkDescriptorSetLayout> layouts(numberPaths, jointDescriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = numberPaths;
    allocInfo.pSetLayouts = layouts.data();
    jointDescriptorSets.resize(numberPaths);
    if (vkAllocateDescriptorSets(device, &allocInfo, jointDescriptorSets.data()) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate joint descriptor sets!");
    }

    const auto& depthSource = occlusionDepth ? occlusionDepth : joints;
    for (uint32_t pathId = 0; pathId < numberPaths; pathId++) {
        VkDescriptorBufferInfo bufferInfos[2]{};
        bufferInfos[0] = {joints->getVkBuffer(pathId), 0, joints->getBufferMemSize()};
        bufferInfos[1] = {depthSource->getVkBuffer(pathId), 0, depthSource->getBufferMemSize()};

        VkWriteDescriptorSet writes[2]{};
        for (uint32_t i = 0; i < 2; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = jointDescriptorSets[pathId];
            writes[i].dstBinding = i;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].descriptorCount = 1;
            writes[i].pBufferInfo = &bufferInfos[i];
        }
        vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);
    }
}

void OpenPoseSkeletonRenderer::createGraphicsPipeline() {
    auto& device = vulkanContext->getDevice();

    auto vertShaderCode = readFile("shaders/skeleton/openpose.vert.spv");
    auto fragShaderCode = readFile("shaders/skeleton/openpose.frag.spv");
    VkShaderModule vertShaderModule = createShaderModule(vertShaderCode, device);
    VkShaderModule fragShaderModule = createShaderModule(fragShaderCode, device);

    VkPipelineShaderStageCreateInfo shaderStages[2]{};
    shaderStages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    shaderStages[0].module = vertShaderModule;
    shaderStages[0].pName = "main";
    shaderStages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    shaderStages[1].module = fragShaderModule;
    shaderStages[1].pName = "main";

    std::vector<VkDynamicState> dynamicStates = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();

    // No vertex buffer: the vertex shader builds one quad per limb or joint
    // instance from gl_VertexIndex and the joint buffer.
    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisampling.minSampleShading = 1.0f;

    // Opaque, like OpenCV's fills: a later limb or joint covers an earlier one.
    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;

    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(OpenPosePushConstants);

    VkDescriptorSetLayout setLayouts[] = {cameraUBO->getDescriptorSetLayout(), jointDescriptorSetLayout};
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 2;
    pipelineLayoutInfo.pSetLayouts = setLayouts;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create pipeline layout!");
    }

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.renderPass = *renderPass;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineIndex = -1;
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
        throw std::runtime_error("failed to create graphics pipeline!");
    }

    vkDestroyShaderModule(device, fragShaderModule, nullptr);
    vkDestroyShaderModule(device, vertShaderModule, nullptr);
}

void OpenPoseSkeletonRenderer::recordCommandBuffer(VkCommandBuffer commandBuffer, VkFramebuffer framebuffer,
                                                   uint32_t pathId) {
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

    VkViewport viewport{};
    viewport.width = static_cast<float>(targetExtent.width);
    viewport.height = static_cast<float>(targetExtent.height);
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.extent = targetExtent;
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

    auto& cameraDescriptorSets = cameraUBO->getDescriptorSets();
    VkDescriptorSet sets[] = {cameraDescriptorSets[pathId], jointDescriptorSets[pathId]};
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 2, sets, 0, nullptr);

    OpenPosePushConstants pushConstants{};
    pushConstants.resolution = glm::vec2(targetExtent.width, targetExtent.height);
    pushConstants.stickWidth = stickWidth;
    pushConstants.jointRadius = jointRadius;
    pushConstants.occlusionBias = occlusionBias;
    pushConstants.useOcclusion = occlusionDepth ? 1u : 0u;
    pushConstants.hideOccludedFace = hideOccludedFace ? 1u : 0u;
    for (uint32_t skeleton = 0; skeleton < skeletonCount; ++skeleton) {
        pushConstants.firstJoint = skeleton * openpose::kJointCount;
        for (uint32_t drawJoints : {0u, 1u}) {
            pushConstants.drawJoints = drawJoints;
            vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(OpenPosePushConstants), &pushConstants);
            vkCmdDraw(commandBuffer, 4, drawJoints ? openpose::kJointCount : openpose::kLimbCount, 0, 0);
        }
    }
}

} // namespace klartraum
