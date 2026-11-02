// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/raster/visibility_pass.hpp"

#include <array>
#include <stdexcept>
#include <string>

#include "klartraum/computegraph/bufferelement.hpp"
#include "klartraum/vulkan_helpers.hpp"

namespace klartraum::raster {

namespace {

constexpr VkFormat kColorFormat = VK_FORMAT_R32_UINT;
constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;
constexpr uint32_t kBufferSlots = 4; // positions, triangles, cameras, view index

void check(VkResult result, const char* operation) {
    if (result != VK_SUCCESS)
        throw std::runtime_error(std::string("VisibilityPass: failed to ") + operation);
}

} // namespace

VisibilityPass::VisibilityPass(VulkanContext& vulkanContext, uint32_t width, uint32_t height, uint32_t triangleCount)
    : context(vulkanContext),
      imageWidth(width),
      imageHeight(height),
      triangleCount(triangleCount) {
    if (width == 0 || height == 0 || triangleCount == 0)
        throw std::invalid_argument("VisibilityPass: image size and triangle count must be positive");
}

VisibilityPass::~VisibilityPass() {
    VkDevice device = context.getDevice();
    for (auto& target : targets) {
        vkDestroyFramebuffer(device, target.framebuffer, nullptr);
        vkDestroyImageView(device, target.colorView, nullptr);
        vkDestroyImageView(device, target.depthView, nullptr);
        vkDestroyImage(device, target.color, nullptr);
        vkDestroyImage(device, target.depth, nullptr);
        vkFreeMemory(device, target.colorMemory, nullptr);
        vkFreeMemory(device, target.depthMemory, nullptr);
    }
    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
    vkDestroyRenderPass(device, renderPass, nullptr);
}

void VisibilityPass::checkInput(ComputeGraphElementPtr input, int index) {
    if (index < 0 || index > 4)
        throw std::runtime_error("VisibilityPass: input index out of range");
    if (dynamic_cast<BufferElementInterface*>(input.get()) == nullptr)
        throw std::runtime_error("VisibilityPass: every input must be a buffer or tensor");
}

void VisibilityPass::createImage(VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect, VkImage& image,
                                 VkDeviceMemory& memory, VkImageView& view) {
    VkDevice device = context.getDevice();
    VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = format;
    info.extent = {imageWidth, imageHeight, 1};
    info.mipLevels = info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = usage;
    context.setSharing(info);
    check(vkCreateImage(device, &info, nullptr, &image), "create an image");
    VkMemoryRequirements requirements{};
    vkGetImageMemoryRequirements(device, image, &requirements);
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex =
        context.findMemoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    check(vkAllocateMemory(device, &allocation, nullptr, &memory), "allocate image memory");
    check(vkBindImageMemory(device, image, memory, 0), "bind image memory");
    VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo.image = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange = {aspect, 0, 1, 0, 1};
    check(vkCreateImageView(device, &viewInfo, nullptr, &view), "create an image view");
}

void VisibilityPass::createRenderPass() {
    std::array<VkAttachmentDescription, 2> attachments{};
    attachments[0].format = kColorFormat;
    attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[0].finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL; // copied into the visibility buffer
    attachments[1] = attachments[0];
    attachments[1].format = kDepthFormat;
    attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorReference{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference depthReference{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorReference;
    subpass.pDepthStencilAttachment = &depthReference;

    // The mesh and cameras were written by earlier commands (e.g. the previous step's optimizer); the copy after
    // the pass reads the colour target.
    std::array<VkSubpassDependency, 2> dependencies{};
    dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[0].dstSubpass = 0;
    dependencies[0].srcStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    dependencies[0].srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    dependencies[0].dstStageMask = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                   VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT |
                                   VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                    VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                    VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dependencies[1].srcSubpass = 0;
    dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependencies[1].dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
    dependencies[1].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

    VkRenderPassCreateInfo info{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    info.attachmentCount = uint32_t(attachments.size());
    info.pAttachments = attachments.data();
    info.subpassCount = 1;
    info.pSubpasses = &subpass;
    info.dependencyCount = uint32_t(dependencies.size());
    info.pDependencies = dependencies.data();
    check(vkCreateRenderPass(context.getDevice(), &info, nullptr, &renderPass), "create the render pass");
}

void VisibilityPass::createPipeline() {
    VkDevice device = context.getDevice();
    std::array<VkDescriptorSetLayoutBinding, kBufferSlots> bindings{};
    for (uint32_t i = 0; i < kBufferSlots; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = kBufferSlots;
    layoutInfo.pBindings = bindings.data();
    check(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout),
          "create the descriptor set layout");

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
    check(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout), "create the pipeline layout");

    VkShaderModule vertex = createShaderModule(readFile("shaders/raster/visibility.vert.spv"), device);
    VkShaderModule fragment = createShaderModule(readFile("shaders/raster/visibility.frag.spv"), device);
    std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
    stages[0].sType = stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertex;
    stages[0].pName = "main";
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragment;
    stages[1].pName = "main";

    VkPipelineVertexInputStateCreateInfo vertexInput{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkViewport viewport{0.0f, 0.0f, float(imageWidth), float(imageHeight), 0.0f, 1.0f};
    VkRect2D scissor{{0, 0}, {imageWidth, imageHeight}};
    VkPipelineViewportStateCreateInfo viewportState{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewportState.viewportCount = viewportState.scissorCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.pScissors = &scissor;
    VkPipelineRasterizationStateCreateInfo rasterization{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rasterization.polygonMode = VK_POLYGON_MODE_FILL;
    rasterization.cullMode = VK_CULL_MODE_NONE; // an optimizer may flip triangles; depth decides
    rasterization.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterization.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    depth.depthTestEnable = depth.depthWriteEnable = VK_TRUE;
    depth.depthCompareOp = VK_COMPARE_OP_LESS;
    VkPipelineColorBlendAttachmentState blendAttachment{};
    blendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT;
    VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blend.attachmentCount = 1;
    blend.pAttachments = &blendAttachment;

    VkGraphicsPipelineCreateInfo info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    info.stageCount = uint32_t(stages.size());
    info.pStages = stages.data();
    info.pVertexInputState = &vertexInput;
    info.pInputAssemblyState = &assembly;
    info.pViewportState = &viewportState;
    info.pRasterizationState = &rasterization;
    info.pMultisampleState = &multisample;
    info.pDepthStencilState = &depth;
    info.pColorBlendState = &blend;
    info.layout = pipelineLayout;
    info.renderPass = renderPass;
    const VkResult result = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline);
    vkDestroyShaderModule(device, vertex, nullptr);
    vkDestroyShaderModule(device, fragment, nullptr);
    check(result, "create the graphics pipeline");
}

void VisibilityPass::_setup(VulkanContext& vulkanContext, uint32_t numberPaths, VkQueue setupQueue) {
    ComputeGraphElement::_setup(vulkanContext, numberPaths, setupQueue);
    VkDevice device = context.getDevice();
    createRenderPass();
    createPipeline();

    VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kBufferSlots * numberPaths};
    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = numberPaths;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    check(vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool), "create the descriptor pool");

    targets.resize(numberPaths);
    for (uint32_t path = 0; path < numberPaths; ++path) {
        Target& target = targets[path];
        createImage(kColorFormat, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                    VK_IMAGE_ASPECT_COLOR_BIT, target.color, target.colorMemory, target.colorView);
        createImage(kDepthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, target.depth,
                    target.depthMemory, target.depthView);
        std::array<VkImageView, 2> views{target.colorView, target.depthView};
        VkFramebufferCreateInfo framebufferInfo{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        framebufferInfo.renderPass = renderPass;
        framebufferInfo.attachmentCount = uint32_t(views.size());
        framebufferInfo.pAttachments = views.data();
        framebufferInfo.width = imageWidth;
        framebufferInfo.height = imageHeight;
        framebufferInfo.layers = 1;
        check(vkCreateFramebuffer(device, &framebufferInfo, nullptr, &target.framebuffer), "create a framebuffer");

        VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocation.descriptorPool = descriptorPool;
        allocation.descriptorSetCount = 1;
        allocation.pSetLayouts = &descriptorSetLayout;
        check(vkAllocateDescriptorSets(device, &allocation, &target.descriptors), "allocate a descriptor set");
        std::array<VkDescriptorBufferInfo, kBufferSlots> bufferInfos{};
        std::array<VkWriteDescriptorSet, kBufferSlots> writes{};
        for (uint32_t i = 0; i < kBufferSlots; ++i) {
            auto* buffer = dynamic_cast<BufferElementInterface*>(getInputElement(int(i)).get());
            bufferInfos[i] = {buffer->getVkBuffer(path), 0, buffer->getBufferMemSize()};
            writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            writes[i].dstSet = target.descriptors;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &bufferInfos[i];
        }
        vkUpdateDescriptorSets(device, kBufferSlots, writes.data(), 0, nullptr);
    }
    auto* visibility = dynamic_cast<BufferElementInterface*>(getInputElement(4).get());
    if (visibility->getBufferMemSize() < size_t(imageWidth) * imageHeight * sizeof(uint32_t))
        throw std::runtime_error("VisibilityPass: the visibility buffer is smaller than the image");
}

void VisibilityPass::_record(VkCommandBuffer commandBuffer, uint32_t pathId) {
    const Target& target = targets.at(pathId);
    std::array<VkClearValue, 2> clears{};
    clears[0].color.uint32[0] = 0; // background
    clears[1].depthStencil = {1.0f, 0};
    VkRenderPassBeginInfo begin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    begin.renderPass = renderPass;
    begin.framebuffer = target.framebuffer;
    begin.renderArea = {{0, 0}, {imageWidth, imageHeight}};
    begin.clearValueCount = uint32_t(clears.size());
    begin.pClearValues = clears.data();
    vkCmdBeginRenderPass(commandBuffer, &begin, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &target.descriptors,
                            0, nullptr);
    vkCmdDraw(commandBuffer, 3 * triangleCount, 1, 0, 0);
    vkCmdEndRenderPass(commandBuffer);

    // The render pass left the target in TRANSFER_SRC_OPTIMAL; the graph's barrier after this element makes the
    // copy visible to the passes that read the visibility buffer.
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {imageWidth, imageHeight, 1};
    auto* visibility = dynamic_cast<BufferElementInterface*>(getInputElement(4).get());
    vkCmdCopyImageToBuffer(commandBuffer, target.color, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           visibility->getVkBuffer(pathId), 1, &region);
}

} // namespace klartraum::raster
