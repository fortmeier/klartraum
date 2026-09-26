#include "klartraum/imgui_overlay.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

#include <imgui.h>
#include <imgui_impl_vulkan.h>

#include "klartraum/vulkan_context.hpp"

namespace klartraum {

namespace {

void checkVkResult(VkResult result) {
    if (result < 0) {
        throw std::runtime_error("ImGui Vulkan backend error: VkResult " + std::to_string(result));
    }
}

// The layout every graph leaves the finished swapchain image in; the overlay
// loads the image in it and hands it on to present/readback unchanged.
VkImageLayout finishedImageLayout(const VulkanContext& vulkanContext) {
    return vulkanContext.hasSurface() ? VK_IMAGE_LAYOUT_PRESENT_SRC_KHR : VK_IMAGE_LAYOUT_GENERAL;
}

}  // namespace

ImGuiOverlay::ImGuiOverlay(VulkanContext& vulkanContext) : vulkanContext_(vulkanContext) {
    IMGUI_CHECKVERSION();
    imguiContext_ = ImGui::CreateContext();
    ImGui::SetCurrentContext(imguiContext_);
    // No imgui.ini: window layouts are not persisted to the working directory.
    ImGui::GetIO().IniFilename = nullptr;
    ImGui::StyleColorsDark();

    auto device = vulkanContext_.getDevice();
    const uint32_t queueFamily = vulkanContext_.getQueueFamilyIndices().graphicsAndComputeFamily.value();

    createRenderPass();

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = queueFamily;
    checkVkResult(vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool_));

    commandBuffers_.resize(BackendConfig::MAX_FRAMES_IN_FLIGHT);
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool_;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = static_cast<uint32_t>(commandBuffers_.size());
    checkVkResult(vkAllocateCommandBuffers(device, &allocInfo, commandBuffers_.data()));

    createSwapChainResources();

    // The backend cycles through ImageCount vertex/index buffer sets, one per
    // RenderDrawData call, so it needs at least one set per frame in flight.
    const uint32_t imageCount = std::max<uint32_t>(vulkanContext_.getNumberOfSwapChainImages(),
                                                   static_cast<uint32_t>(BackendConfig::MAX_FRAMES_IN_FLIGHT));

    ImGui_ImplVulkan_InitInfo initInfo{};
    initInfo.ApiVersion = VK_API_VERSION_1_3;
    initInfo.Instance = vulkanContext_.getInstance();
    initInfo.PhysicalDevice = vulkanContext_.getPhysicalDevice();
    initInfo.Device = device;
    initInfo.QueueFamily = queueFamily;
    initInfo.Queue = vulkanContext_.getGraphicsQueue();
    initInfo.DescriptorPoolSize = IMGUI_IMPL_VULKAN_MINIMUM_SAMPLED_IMAGE_POOL_SIZE * 2;
    initInfo.MinImageCount = 2;
    initInfo.ImageCount = imageCount;
    initInfo.PipelineInfoMain.RenderPass = renderPass_;
    initInfo.PipelineInfoMain.Subpass = 0;
    initInfo.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    initInfo.CheckVkResultFn = checkVkResult;
    if (!ImGui_ImplVulkan_Init(&initInfo)) {
        throw std::runtime_error("failed to initialize the ImGui Vulkan backend!");
    }
}

ImGuiOverlay::~ImGuiOverlay() {
    auto device = vulkanContext_.getDevice();
    vkDeviceWaitIdle(device);

    ImGui::SetCurrentContext(imguiContext_);
    ImGui_ImplVulkan_Shutdown();
    ImGui::DestroyContext(imguiContext_);

    destroySwapChainResources();
    vkDestroyCommandPool(device, commandPool_, nullptr);
    vkDestroyRenderPass(device, renderPass_, nullptr);
}

void ImGuiOverlay::createRenderPass() {
    const VkImageLayout layout = finishedImageLayout(vulkanContext_);

    format_ = vulkanContext_.getSwapChainImageFormat();

    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = format_;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    // Draw on top of the frame the graphs rendered.
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = layout;
    colorAttachment.finalLayout = layout;

    VkAttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;

    // The wait semaphore orders the graphs' writes (color attachment or
    // compute storage writes) before this stage; the dependency covers the
    // load and the transition into COLOR_ATTACHMENT_OPTIMAL.
    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments = &colorAttachment;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    checkVkResult(vkCreateRenderPass(vulkanContext_.getDevice(), &renderPassInfo, nullptr, &renderPass_));
}

void ImGuiOverlay::createSwapChainResources() {
    auto device = vulkanContext_.getDevice();
    const uint32_t numImages = vulkanContext_.getNumberOfSwapChainImages();
    const VkExtent2D extent = vulkanContext_.getSwapChainExtent();

    framebuffers_.resize(numImages);
    finishedSemaphores_.resize(numImages);
    for (uint32_t i = 0; i < numImages; ++i) {
        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = renderPass_;
        framebufferInfo.attachmentCount = 1;
        framebufferInfo.pAttachments = &vulkanContext_.getImageView(i);
        framebufferInfo.width = extent.width;
        framebufferInfo.height = extent.height;
        framebufferInfo.layers = 1;
        checkVkResult(vkCreateFramebuffer(device, &framebufferInfo, nullptr, &framebuffers_[i]));

        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        checkVkResult(vkCreateSemaphore(device, &semaphoreInfo, nullptr, &finishedSemaphores_[i]));
    }
}

void ImGuiOverlay::destroySwapChainResources() {
    auto device = vulkanContext_.getDevice();
    for (auto framebuffer : framebuffers_) {
        vkDestroyFramebuffer(device, framebuffer, nullptr);
    }
    for (auto semaphore : finishedSemaphores_) {
        vkDestroySemaphore(device, semaphore, nullptr);
    }
    framebuffers_.clear();
    finishedSemaphores_.clear();
}

void ImGuiOverlay::onSwapChainRecreated() {
    // The render pass, and the backend pipeline built against it, are tied
    // to the swapchain format; only the images may change.
    if (vulkanContext_.getSwapChainImageFormat() != format_) {
        throw std::runtime_error("ImGuiOverlay: the swapchain format changed on recreation!");
    }
    vkDeviceWaitIdle(vulkanContext_.getDevice());
    destroySwapChainResources();
    createSwapChainResources();
}

void ImGuiOverlay::newFrame() {
    ImGui::SetCurrentContext(imguiContext_);
    ImGui_ImplVulkan_NewFrame();
    ImGui::NewFrame();
}

void ImGuiOverlay::render() {
    ImGui::SetCurrentContext(imguiContext_);
    ImGui::Render();
}

VkSemaphore ImGuiOverlay::submit(VkQueue queue, uint32_t imageIndex, uint32_t frameIndex,
                                 VkSemaphore waitSemaphore, VkFence fence) {
    ImGui::SetCurrentContext(imguiContext_);

    VkCommandBuffer cmd = commandBuffers_[frameIndex];
    checkVkResult(vkResetCommandBuffer(cmd, 0));

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    checkVkResult(vkBeginCommandBuffer(cmd, &beginInfo));

    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = renderPass_;
    renderPassInfo.framebuffer = framebuffers_[imageIndex];
    renderPassInfo.renderArea.extent = vulkanContext_.getSwapChainExtent();
    vkCmdBeginRenderPass(cmd, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

    // Before the first render() there is no draw data; the pass then only
    // passes the image through.
    if (ImDrawData* drawData = ImGui::GetDrawData()) {
        ImGui_ImplVulkan_RenderDrawData(drawData, cmd);
    }

    vkCmdEndRenderPass(cmd);
    checkVkResult(vkEndCommandBuffer(cmd));

    const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = &waitSemaphore;
    submitInfo.pWaitDstStageMask = &waitStage;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &finishedSemaphores_[imageIndex];
    checkVkResult(vkQueueSubmit(queue, 1, &submitInfo, fence));

    return finishedSemaphores_[imageIndex];
}

}  // namespace klartraum
