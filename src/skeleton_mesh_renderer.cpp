// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/skeleton_mesh_renderer.hpp"

#include <cmath>
#include <set>
#include <stdexcept>

#include "klartraum/vulkan_helpers.hpp"

namespace klartraum {

namespace {

constexpr uint32_t kSegments = 16; // around a cylinder and a sphere
constexpr uint32_t kRings = 8;     // from pole to pole of a sphere

// Unit shapes as triangle lists, read by the vertex shader: an open cylinder of radius 1 from y = 0 to y = 1, then a
// sphere of radius 1. A vertex's normal follows from its position.
std::vector<glm::vec4> cylinderTriangles() {
    std::vector<glm::vec4> vertices;
    const float step = 2.0f * float(M_PI) / float(kSegments);
    for (uint32_t i = 0; i < kSegments; ++i) {
        const glm::vec4 a(std::cos(step * i), 0.0f, std::sin(step * i), 1.0f);
        const glm::vec4 b(std::cos(step * (i + 1)), 0.0f, std::sin(step * (i + 1)), 1.0f);
        const glm::vec4 up(0.0f, 1.0f, 0.0f, 0.0f);
        vertices.insert(vertices.end(), {a, b, b + up, a, b + up, a + up});
    }
    return vertices;
}

std::vector<glm::vec4> sphereTriangles() {
    auto point = [](uint32_t ring, uint32_t segment) {
        const float polar = float(M_PI) * float(ring) / float(kRings);
        const float azimuth = 2.0f * float(M_PI) * float(segment) / float(kSegments);
        return glm::vec4(std::sin(polar) * std::cos(azimuth), std::cos(polar), std::sin(polar) * std::sin(azimuth),
                         1.0f);
    };
    std::vector<glm::vec4> vertices;
    for (uint32_t ring = 0; ring < kRings; ++ring) {
        for (uint32_t segment = 0; segment < kSegments; ++segment) {
            const glm::vec4 a = point(ring, segment), b = point(ring, segment + 1);
            const glm::vec4 c = point(ring + 1, segment + 1), d = point(ring + 1, segment);
            vertices.insert(vertices.end(), {a, b, c, a, c, d});
        }
    }
    return vertices;
}

} // namespace

SkeletonMeshRenderer::SkeletonMeshRenderer(std::shared_ptr<BufferElementInterface> joints,
                                           std::vector<std::pair<uint32_t, uint32_t>> bones,
                                           std::shared_ptr<BufferElementInterface> params, VkExtent2D targetExtent)
    : joints(std::move(joints)),
      params(std::move(params)),
      targetExtent(targetExtent),
      boneCount(uint32_t(bones.size())) {
    if (bones.empty()) {
        throw std::invalid_argument("SkeletonMeshRenderer: needs at least one bone");
    }
    std::set<uint32_t> used;
    for (const auto& [a, b] : bones) {
        indices.push_back(a);
        indices.push_back(b);
        used.insert(a);
        used.insert(b);
    }
    indices.insert(indices.end(), used.begin(), used.end());
    sphereCount = uint32_t(used.size());
}

SkeletonMeshRenderer::~SkeletonMeshRenderer() {
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

void SkeletonMeshRenderer::setOcclusionDepth(std::shared_ptr<BufferElementInterface> depth, float relativeBias) {
    if (!depth) {
        throw std::runtime_error("SkeletonMeshRenderer: the occlusion depth buffer is missing");
    }
    occlusionDepth = std::move(depth);
    occlusionBias = relativeBias;
}

void SkeletonMeshRenderer::initialize(VulkanContext& vulkanContext, VkRenderPass& renderPass,
                                      std::shared_ptr<CameraUboType> cameraUBO) {
    DrawComponent::initialize(vulkanContext, renderPass, cameraUBO);
    if (params->getBufferMemSize() < sizeof(SkeletonGaussianParams)) {
        throw std::runtime_error("SkeletonMeshRenderer: the parameter buffer is too small");
    }
    if (occlusionDepth &&
        occlusionDepth->getBufferMemSize() < sizeof(float) * targetExtent.width * targetExtent.height) {
        throw std::runtime_error("SkeletonMeshRenderer: the occlusion depth buffer is smaller than the target");
    }
    indexBuffer = std::make_unique<VulkanBuffer<uint32_t>>(vulkanContext, uint32_t(indices.size()));
    indexBuffer->memcopyFrom(indices);
    auto mesh = cylinderTriangles();
    cylinderVertexCount = uint32_t(mesh.size());
    const auto sphere = sphereTriangles();
    sphereVertexCount = uint32_t(sphere.size());
    mesh.insert(mesh.end(), sphere.begin(), sphere.end());
    meshBuffer = std::make_unique<VulkanBuffer<glm::vec4>>(vulkanContext, uint32_t(mesh.size()));
    meshBuffer->memcopyFrom(mesh);
    createDescriptorSets();
    createGraphicsPipeline();
}

void SkeletonMeshRenderer::createDescriptorSets() {
    auto& device = vulkanContext->getDevice();

    // 0: joints, 1: indices, 2: parameters, 3: mesh (vertex shader); 4: occlusion depth (fragment shader; the joints
    // again when there is none, then it is not read).
    constexpr uint32_t kBindings = 5;
    VkDescriptorSetLayoutBinding bindings[kBindings]{};
    for (uint32_t i = 0; i < kBindings; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorCount = 1;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].stageFlags = i == 4 ? VK_SHADER_STAGE_FRAGMENT_BIT : VK_SHADER_STAGE_VERTEX_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = kBindings;
    layoutInfo.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("SkeletonMeshRenderer: failed to create the descriptor set layout");
    }

    VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kBindings * numberPaths};
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    poolInfo.maxSets = numberPaths;
    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("SkeletonMeshRenderer: failed to create the descriptor pool");
    }

    std::vector<VkDescriptorSetLayout> layouts(numberPaths, descriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = numberPaths;
    allocInfo.pSetLayouts = layouts.data();
    descriptorSets.resize(numberPaths);
    if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()) != VK_SUCCESS) {
        throw std::runtime_error("SkeletonMeshRenderer: failed to allocate the descriptor sets");
    }

    const auto& depthSource = occlusionDepth ? occlusionDepth : joints;
    for (uint32_t pathId = 0; pathId < numberPaths; pathId++) {
        VkDescriptorBufferInfo bufferInfos[kBindings] = {
            {joints->getVkBuffer(pathId), 0, joints->getBufferMemSize()},
            {indexBuffer->getBuffer(), 0, indexBuffer->getBufferMemSize()},
            {params->getVkBuffer(pathId), 0, params->getBufferMemSize()},
            {meshBuffer->getBuffer(), 0, meshBuffer->getBufferMemSize()},
            {depthSource->getVkBuffer(pathId), 0, depthSource->getBufferMemSize()},
        };
        VkWriteDescriptorSet writes[kBindings]{};
        for (uint32_t i = 0; i < kBindings; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = descriptorSets[pathId];
            writes[i].dstBinding = i;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].descriptorCount = 1;
            writes[i].pBufferInfo = &bufferInfos[i];
        }
        vkUpdateDescriptorSets(device, kBindings, writes, 0, nullptr);
    }
}

void SkeletonMeshRenderer::createGraphicsPipeline() {
    auto& device = vulkanContext->getDevice();

    auto vertShaderCode = readFile("shaders/skeleton/skeleton_mesh.vert.spv");
    auto fragShaderCode = readFile("shaders/skeleton/skeleton_mesh.frag.spv");
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

    // No vertex buffer: the vertex shader reads the unit shapes from a storage buffer.
    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    // Both faces: the camera's projection may mirror the image.
    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    auto depthStencil = pipelineDepthState();

    // Blended by the skeleton's opacity over the image.
    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_TRUE;
    colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
    colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;

    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pushConstantRange.size = sizeof(SkeletonMeshPushConstants);

    VkDescriptorSetLayout setLayouts[] = {cameraUBO->getDescriptorSetLayout(), descriptorSetLayout};
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 2;
    pipelineLayoutInfo.pSetLayouts = setLayouts;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("SkeletonMeshRenderer: failed to create the pipeline layout");
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
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.renderPass = *renderPass;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineIndex = -1;
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
        throw std::runtime_error("SkeletonMeshRenderer: failed to create the graphics pipeline");
    }

    vkDestroyShaderModule(device, fragShaderModule, nullptr);
    vkDestroyShaderModule(device, vertShaderModule, nullptr);
}

void SkeletonMeshRenderer::recordCommandBuffer(VkCommandBuffer commandBuffer, VkFramebuffer framebuffer,
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

    VkDescriptorSet sets[] = {cameraUBO->getDescriptorSets()[pathId], descriptorSets[pathId]};
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 2, sets, 0, nullptr);

    SkeletonMeshPushConstants pushConstants{};
    pushConstants.resolution = glm::vec2(targetExtent.width, targetExtent.height);
    pushConstants.boneCount = boneCount;
    pushConstants.occlusionBias = occlusionBias;
    pushConstants.useOcclusion = occlusionDepth ? 1u : 0u;
    for (uint32_t drawSpheres : {0u, 1u}) {
        pushConstants.drawSpheres = drawSpheres;
        vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                           sizeof(SkeletonMeshPushConstants), &pushConstants);
        if (drawSpheres) {
            vkCmdDraw(commandBuffer, sphereVertexCount, sphereCount, cylinderVertexCount, 0);
        } else {
            vkCmdDraw(commandBuffer, cylinderVertexCount, boneCount, 0, 0);
        }
    }
}

} // namespace klartraum
