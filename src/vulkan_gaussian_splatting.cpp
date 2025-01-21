#include "klartraum/vulkan_gaussian_splatting.hpp"
#include <stdexcept>

namespace klartraum {

VulkanGaussianSplatting::VulkanGaussianSplatting(BackendVulkan &backendVulkan) : backendVulkan(backendVulkan) {
    createCommandPool();
    createCommandBuffers();
    createSyncObjects();
}

VulkanGaussianSplatting::~VulkanGaussianSplatting() {
    auto device = backendVulkan.getDevice();
    for (size_t i = 0; i < backendVulkan.MAX_FRAMES_IN_FLIGHT; i++)
    {
        vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
    }

    vkDestroyCommandPool(device, commandPool, nullptr);
}

VkSemaphore VulkanGaussianSplatting::draw(uint32_t currentFrame, VkFramebuffer framebuffer, VkSemaphore imageAvailableSemaphore) {
    auto device = backendVulkan.getDevice();
    auto swapChain = backendVulkan.getSwapChain();
    auto graphicsQueue = backendVulkan.getGraphicsQueue();



    vkResetCommandBuffer(commandBuffers[currentFrame], 0);
    recordCommandBuffer(commandBuffers[currentFrame], framebuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    VkSemaphore waitSemaphores[] = { imageAvailableSemaphore };
    VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;

    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffers[currentFrame];

    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &renderFinishedSemaphores[currentFrame];

    if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, nullptr) != VK_SUCCESS) {
        throw std::runtime_error("failed to submit draw command buffer!");
    }

   return renderFinishedSemaphores[currentFrame];
}

void VulkanGaussianSplatting::createCommandPool() {

    auto device = backendVulkan.getDevice();

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = backendVulkan.getQueueFamilyIndices().graphicsFamily.value();

    if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create command pool!");
    }    
}

void VulkanGaussianSplatting::createCommandBuffers() {
    auto device = backendVulkan.getDevice();

    commandBuffers.resize(BackendVulkan::MAX_FRAMES_IN_FLIGHT);
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = commandBuffers.size();

    if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate command buffers!");
    }
}

void VulkanGaussianSplatting::createSyncObjects()
{
    auto device = backendVulkan.getDevice();

    renderFinishedSemaphores.resize(backendVulkan.MAX_FRAMES_IN_FLIGHT);

    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    for (size_t i = 0; i < backendVulkan.MAX_FRAMES_IN_FLIGHT; i++)
    {
        if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create semaphores!");
        }
    }    
}

void VulkanGaussianSplatting::recordCommandBuffer(VkCommandBuffer commandBuffer, VkFramebuffer framebuffer)
{
    auto swapChainExtent = backendVulkan.getSwapChainExtent();
    auto graphicsPipeline = backendVulkan.getGraphicsPipeline();
    
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = 0; // Optional
    beginInfo.pInheritanceInfo = nullptr; // Optional

    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
        throw std::runtime_error("failed to begin recording command buffer!");
    }

    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = backendVulkan.getRenderPass();
    renderPassInfo.framebuffer = framebuffer;

    renderPassInfo.renderArea.offset = { 0, 0 };
    renderPassInfo.renderArea.extent = swapChainExtent;

    VkClearValue clearColor = { {{0.0f, 0.0f, 0.0f, 1.0f}} };
    renderPassInfo.clearValueCount = 1;
    renderPassInfo.pClearValues = &clearColor;

    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(swapChainExtent.width);
    viewport.height = static_cast<float>(swapChainExtent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset = { 0, 0 };
    scissor.extent = swapChainExtent;
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

    vkCmdDraw(commandBuffer, 3, 1, 0, 0);

    vkCmdEndRenderPass(commandBuffer);

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to record command buffer!");
    }
}

} // namespace klartraum
