#include <stdexcept>

#include "klartraum/window.hpp"
#include "klartraum/vulkan_context.hpp"

namespace klartraum {

Window::Window(VulkanContext& vulkanContext)
    : vulkanContext_(vulkanContext) {}

Window::~Window() {
    auto& device = vulkanContext_.getDevice();
    for (auto sem : compositeFinished_) {
        if (sem != VK_NULL_HANDLE) vkDestroySemaphore(device, sem, nullptr);
    }
    if (!composite_.empty()) {
        vkFreeCommandBuffers(device, vulkanContext_.getCommandPool(),
                             (uint32_t)composite_.size(), composite_.data());
    }
}

std::shared_ptr<ImageViewSrc> Window::makeViewport(int x, int y, uint32_t width, uint32_t height) {
    if (finalized_) {
        throw std::runtime_error("Window::makeViewport called after the composite was built!");
    }
    uint32_t numImages = vulkanContext_.getNumberOfSwapChainImages();
    auto target = std::make_shared<OffscreenTarget>(
        vulkanContext_, VkExtent2D{ width, height }, numImages);

    VkRect2D rect{};
    rect.offset = { x, y };
    rect.extent = { width, height };
    viewports_.push_back({ target, rect });
    return target;
}

void Window::finalize() {
    if (finalized_) return;
    finalized_ = true;

    auto& device = vulkanContext_.getDevice();
    uint32_t numImages = vulkanContext_.getNumberOfSwapChainImages();

    // Headless has no surface to present to; mirror the existing convention that
    // leaves the final swapchain image in GENERAL so readback works.
    const VkImageLayout finalSwapLayout = vulkanContext_.hasSurface()
        ? VK_IMAGE_LAYOUT_PRESENT_SRC_KHR : VK_IMAGE_LAYOUT_GENERAL;

    composite_.resize(numImages);
    compositeFinished_.resize(numImages);

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool        = vulkanContext_.getCommandPool();
    allocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = numImages;
    if (vkAllocateCommandBuffers(device, &allocInfo, composite_.data()) != VK_SUCCESS) {
        throw std::runtime_error("Window: failed to allocate composite command buffers!");
    }

    VkSemaphoreCreateInfo semInfo{};
    semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkImageSubresourceRange fullRange{};
    fullRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    fullRange.levelCount = 1;
    fullRange.layerCount = 1;

    for (uint32_t p = 0; p < numImages; ++p) {
        if (vkCreateSemaphore(device, &semInfo, nullptr, &compositeFinished_[p]) != VK_SUCCESS) {
            throw std::runtime_error("Window: failed to create composite semaphore!");
        }

        VkCommandBuffer cmd = composite_[p];
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vkBeginCommandBuffer(cmd, &beginInfo);

        VkImage swapImage = vulkanContext_.getSwapChainImage(p);

        // Swapchain image -> TRANSFER_DST so we can clear and blit into it.
        VkImageMemoryBarrier toDst{};
        toDst.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        toDst.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
        toDst.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toDst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toDst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toDst.image               = swapImage;
        toDst.subresourceRange    = fullRange;
        toDst.srcAccessMask       = 0;
        toDst.dstAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &toDst);

        // Clear the whole image so any region not covered by a viewport is black.
        VkClearColorValue black{};
        black.float32[0] = 0.0f; black.float32[1] = 0.0f;
        black.float32[2] = 0.0f; black.float32[3] = 1.0f;
        vkCmdClearColorImage(cmd, swapImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             &black, 1, &fullRange);

        // Order the clear's writes before the blits' writes (same image).
        VkImageMemoryBarrier clearToBlit{};
        clearToBlit.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        clearToBlit.oldLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        clearToBlit.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        clearToBlit.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        clearToBlit.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        clearToBlit.image               = swapImage;
        clearToBlit.subresourceRange    = fullRange;
        clearToBlit.srcAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
        clearToBlit.dstAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &clearToBlit);

        // The viewport scene left each offscreen image in TRANSFER_SRC_OPTIMAL
        // (OffscreenTarget::getFinalLayoutOverride); the graphs' finished
        // semaphores (awaited at submit) make those writes visible here.
        for (auto& vp : viewports_) {
            VkImageBlit blit{};
            blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            blit.srcSubresource.layerCount = 1;
            blit.srcOffsets[0] = { 0, 0, 0 };
            blit.srcOffsets[1] = { (int32_t)vp.rect.extent.width,
                                   (int32_t)vp.rect.extent.height, 1 };
            blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            blit.dstSubresource.layerCount = 1;
            blit.dstOffsets[0] = { vp.rect.offset.x, vp.rect.offset.y, 0 };
            blit.dstOffsets[1] = { vp.rect.offset.x + (int32_t)vp.rect.extent.width,
                                   vp.rect.offset.y + (int32_t)vp.rect.extent.height, 1 };
            vkCmdBlitImage(cmd,
                vp.target->getImage(p), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                swapImage,             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1, &blit, VK_FILTER_NEAREST);
        }

        // Swapchain image -> final layout for present (or readback when headless).
        VkImageMemoryBarrier toFinal{};
        toFinal.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        toFinal.oldLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toFinal.newLayout           = finalSwapLayout;
        toFinal.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toFinal.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toFinal.image               = swapImage;
        toFinal.subresourceRange    = fullRange;
        toFinal.srcAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
        toFinal.dstAccessMask       = VK_ACCESS_MEMORY_READ_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            0, 0, nullptr, 0, nullptr, 1, &toFinal);

        vkEndCommandBuffer(cmd);
    }
}

VkSemaphore Window::submitComposite(VkQueue queue, uint32_t imageIndex,
                                    const std::vector<VkSemaphore>& waitSemaphores,
                                    VkFence fence) {
    finalize();
    if (imageIndex >= composite_.size()) {
        throw std::runtime_error("Window::submitComposite: imageIndex out of range!");
    }

    std::vector<VkPipelineStageFlags> waitStages(
        waitSemaphores.size(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

    VkSubmitInfo si{};
    si.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount   = 1;
    si.pCommandBuffers      = &composite_[imageIndex];
    si.waitSemaphoreCount   = (uint32_t)waitSemaphores.size();
    si.pWaitSemaphores      = waitSemaphores.data();
    si.pWaitDstStageMask    = waitStages.data();
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores    = &compositeFinished_[imageIndex];

    if (vkQueueSubmit(queue, 1, &si, fence) != VK_SUCCESS) {
        throw std::runtime_error("Window::submitComposite: failed to submit!");
    }
    return compositeFinished_[imageIndex];
}

} // namespace klartraum
