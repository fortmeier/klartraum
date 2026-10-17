// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include <stdexcept>

#include "klartraum/character_mask.hpp"
#include "klartraum/openpose_skeleton_renderer.hpp"
#include "klartraum/vulkan_helpers.hpp"

namespace klartraum {

namespace {

struct SceneMaskPushConstants {
    uint32_t width;
    uint32_t height;
};

} // namespace

const std::vector<CharacterMaskSegment>& characterMaskSegments() {
    using namespace openpose;
    // Must match SEGMENT_JOINTS and SEGMENT_RADII in shaders/skeleton/character_mask.vert.
    static const std::vector<CharacterMaskSegment> segments = {
        {Neck, RightShoulder, 0.07f},
        {Neck, LeftShoulder, 0.07f},
        {RightShoulder, RightElbow, 0.07f},
        {RightElbow, RightWrist, 0.055f},
        {LeftShoulder, LeftElbow, 0.07f},
        {LeftElbow, LeftWrist, 0.055f},
        {Neck, RightHip, 0.14f},
        {RightHip, RightKnee, 0.085f},
        {RightKnee, RightAnkle, 0.065f},
        {Neck, LeftHip, 0.14f},
        {LeftHip, LeftKnee, 0.085f},
        {LeftKnee, LeftAnkle, 0.065f},
        {Neck, Nose, 0.06f},
        {Nose, RightEye, 0.05f},
        {RightEye, RightEar, 0.05f},
        {Nose, LeftEye, 0.05f},
        {LeftEye, LeftEar, 0.05f},
        // The sides of the torso, the hips, and the head: a disc around the middle of the ears.
        {RightShoulder, RightHip, 0.11f},
        {LeftShoulder, LeftHip, 0.11f},
        {RightHip, LeftHip, 0.11f},
        {RightEar, LeftEar, 0.1f, true},
    };
    return segments;
}

CharacterMaskRenderer::CharacterMaskRenderer(std::shared_ptr<BufferElementInterface> joints, VkExtent2D targetExtent)
    : joints(std::move(joints)),
      targetExtent(targetExtent) {}

CharacterMaskRenderer::~CharacterMaskRenderer() {
    // Nothing was created if initialize() never ran.
    if (vulkanContext == nullptr) {
        return;
    }
    auto& device = vulkanContext->getDevice();
    vkDestroyPipeline(device, graphicsPipeline, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
}

void CharacterMaskRenderer::setOcclusionDepth(std::shared_ptr<BufferElementInterface> depth, float relativeBias) {
    if (!depth) {
        throw std::runtime_error("CharacterMaskRenderer: the occlusion depth buffer is missing");
    }
    occlusionDepth = std::move(depth);
    occlusionBias = relativeBias;
}

void CharacterMaskRenderer::initialize(VulkanContext& vulkanContext, VkRenderPass& renderPass,
                                       std::shared_ptr<CameraUboType> cameraUBO) {
    DrawComponent::initialize(vulkanContext, renderPass, cameraUBO);
    if (joints->getBufferMemSize() < sizeof(glm::vec4) * openpose::kJointCount) {
        throw std::runtime_error("CharacterMaskRenderer: the joint buffer is smaller than a skeleton");
    }
    if (occlusionDepth &&
        occlusionDepth->getBufferMemSize() < sizeof(float) * targetExtent.width * targetExtent.height) {
        throw std::runtime_error("CharacterMaskRenderer: the occlusion depth buffer is smaller than the target");
    }
    createDescriptorSets();
    createGraphicsPipeline();
}

void CharacterMaskRenderer::createDescriptorSets() {
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
    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create character mask descriptor set layout!");
    }

    VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 * numberPaths};
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    poolInfo.maxSets = numberPaths;
    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create character mask descriptor pool!");
    }

    std::vector<VkDescriptorSetLayout> layouts(numberPaths, descriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = numberPaths;
    allocInfo.pSetLayouts = layouts.data();
    descriptorSets.resize(numberPaths);
    if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate character mask descriptor sets!");
    }

    const auto& depthSource = occlusionDepth ? occlusionDepth : joints;
    for (uint32_t pathId = 0; pathId < numberPaths; pathId++) {
        VkDescriptorBufferInfo bufferInfos[2]{};
        bufferInfos[0] = {joints->getVkBuffer(pathId), 0, joints->getBufferMemSize()};
        bufferInfos[1] = {depthSource->getVkBuffer(pathId), 0, depthSource->getBufferMemSize()};
        VkWriteDescriptorSet writes[2]{};
        for (uint32_t i = 0; i < 2; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = descriptorSets[pathId];
            writes[i].dstBinding = i;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].descriptorCount = 1;
            writes[i].pBufferInfo = &bufferInfos[i];
        }
        vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);
    }
}

void CharacterMaskRenderer::createGraphicsPipeline() {
    auto& device = vulkanContext->getDevice();

    auto vertShaderCode = readFile("shaders/skeleton/character_mask.vert.spv");
    auto fragShaderCode = readFile("shaders/skeleton/character_mask.frag.spv");
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

    // No vertex buffer: the vertex shader builds one quad per segment from gl_VertexIndex and the joint buffer.
    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;

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
    pushConstantRange.size = sizeof(CharacterMaskPushConstants);

    VkDescriptorSetLayout setLayouts[] = {cameraUBO->getDescriptorSetLayout(), descriptorSetLayout};
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
    pipelineInfo.basePipelineIndex = -1;
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
        throw std::runtime_error("failed to create graphics pipeline!");
    }

    vkDestroyShaderModule(device, fragShaderModule, nullptr);
    vkDestroyShaderModule(device, vertShaderModule, nullptr);
}

void CharacterMaskRenderer::recordCommandBuffer(VkCommandBuffer commandBuffer, VkFramebuffer framebuffer,
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
    VkDescriptorSet sets[] = {cameraDescriptorSets[pathId], descriptorSets[pathId]};
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 2, sets, 0, nullptr);

    CharacterMaskPushConstants pushConstants{};
    pushConstants.resolution = glm::vec2(targetExtent.width, targetExtent.height);
    pushConstants.margin = margin;
    pushConstants.occlusionBias = occlusionBias;
    pushConstants.useOcclusion = occlusionDepth ? 1u : 0u;
    vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                       sizeof(CharacterMaskPushConstants), &pushConstants);
    vkCmdDraw(commandBuffer, 4, static_cast<uint32_t>(characterMaskSegments().size()), 0, 0);
}

std::shared_ptr<ComputeGraphElement> createSceneMask(VulkanContext& vulkanContext,
                                                     std::shared_ptr<BufferElementInterface> depth,
                                                     std::shared_ptr<ImageViewSrc> target,
                                                     ComputeGraphElementPtr producer) {
    // The depth buffer is allocated only when the graph is compiled, so its size cannot be checked here.
    const VkExtent2D extent = target->getImageExtent(0);
    auto mask = vulkanContext.create<GeneralComputation<SceneMaskPushConstants>>("shaders/image/scene_mask.comp.spv");
    mask->setName("SceneMask");
    mask->setPushConstants({{extent.width, extent.height}});
    mask->setGroupCount((extent.width + 7) / 8, (extent.height + 7) / 8, 1);
    mask->setInput(depth, 0);
    mask->setInput(target, 1);
    // Every pixel is written, so the previous contents need not be kept.
    mask->setImageLayoutTransition(1, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                                   VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                                   VK_ACCESS_SHADER_WRITE_BIT);
    if (producer) {
        mask->addDependency(producer);
    }
    return mask;
}

} // namespace klartraum
