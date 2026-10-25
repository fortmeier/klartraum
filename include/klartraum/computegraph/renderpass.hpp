// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_COMPUTEGRAPH_RENDERPASS_HPP
#define KLARTRAUM_COMPUTEGRAPH_RENDERPASS_HPP

#include <map>
#include <array>
#include <algorithm>
#include <vector>
#include <vulkan/vulkan.h>
#include "klartraum/computegraph/imageviewsrc.hpp"
#include "klartraum/computegraph/rendergraphelement.hpp"
#include "klartraum/draw_component.hpp"

namespace klartraum {

class RenderPass : public ImageViewSrc, public RenderGraphElement {
public:
    RenderPass(VkFormat swapChainImageFormat, VkExtent2D extent)
        : swapChainImageFormat(swapChainImageFormat),
          swapChainExtent(extent) {

          };

    ~RenderPass() {
        // Nothing was created if the pass was never compiled into a graph.
        if (vulkanContext == nullptr) {
            return;
        }
        auto& device = vulkanContext->getDevice();
        for (auto framebuffer : framebuffers) {
            vkDestroyFramebuffer(device, framebuffer, nullptr);
        }
        vkDestroyRenderPass(device, renderPass, nullptr);
    };

    virtual void checkInput(ComputeGraphElementPtr input, int index = 0) {
        ImageViewSrc* imageViewSrc = std::dynamic_pointer_cast<ImageViewSrc>(input).get();
        if (index == 0 && imageViewSrc == nullptr) {
            throw std::runtime_error("input is not an ImageViewSrc!");
        }
        CameraUboType* cameraUbo = std::dynamic_pointer_cast<CameraUboType>(input).get();
        if (index == 1 && cameraUbo == nullptr) {
            throw std::runtime_error("input is not a CameraUboType!");
        }
        if (index > 1) {
            throw std::runtime_error("input index out of range!");
        }
    }

    virtual const char* getType() const { return "RenderPass"; }

    virtual void _setup(VulkanContext& vulkanContext, uint32_t numberPaths, VkQueue setupQueue = VK_NULL_HANDLE) {
        this->vulkanContext = &vulkanContext;

        auto& device = vulkanContext.getDevice();
        if (depthAttachment && (depthAttachment->extent().width != swapChainExtent.width ||
                                depthAttachment->extent().height != swapChainExtent.height)) {
            throw std::invalid_argument("RenderPass: depth and color extents differ");
        }

        VkAttachmentDescription colorAttachment{};
        colorAttachment.format = swapChainImageFormat;
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;

        colorAttachment.loadOp = loadExisting ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

        // For windowed rendering the image must be in PRESENT_SRC_KHR after
        // the renderpass so vkQueuePresentKHR accepts it.  For headless
        // offscreen rendering GENERAL is fine. A viewport target overrides this
        // with TRANSFER_SRC_OPTIMAL so the Window composite can blit from it.
        // Drawing over the image needs its contents, which its source left in
        // GENERAL (see ImageViewSrc).
        colorAttachment.initialLayout = loadExisting ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED;
        VkImageLayout defaultFinal =
            vulkanContext.hasSurface() ? VK_IMAGE_LAYOUT_PRESENT_SRC_KHR : VK_IMAGE_LAYOUT_GENERAL;
        if (auto* ivs = std::dynamic_pointer_cast<ImageViewSrcInterface>(getInputElement(0)).get()) {
            colorAttachment.finalLayout = ivs->getFinalLayoutOverride().value_or(defaultFinal);
        } else {
            colorAttachment.finalLayout = defaultFinal;
        }
        outputLayout = colorAttachment.finalLayout;

        VkAttachmentReference colorAttachmentRef{};
        colorAttachmentRef.attachment = 0;
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;

        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;

        VkAttachmentDescription depthDescription{};
        depthDescription.format = DepthAttachment::format();
        depthDescription.samples = VK_SAMPLE_COUNT_1_BIT;
        depthDescription.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        depthDescription.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        depthDescription.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depthDescription.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthDescription.initialLayout = depthDescription.finalLayout =
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        VkAttachmentReference depthReference{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        if (depthAttachment)
            subpass.pDepthStencilAttachment = &depthReference;
        std::array<VkAttachmentDescription, 2> attachments{colorAttachment, depthDescription};

        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = depthAttachment ? 2u : 1u;
        renderPassInfo.pAttachments = attachments.data();
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;

        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        // Loading the contents reads the attachment after its layout
        // transition, so the transition must be made visible to reads, too.
        dependency.dstAccessMask = loadExisting
                                       ? VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                                       : VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        if (depthAttachment) {
            dependency.srcStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
            dependency.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
            dependency.dstStageMask |=
                VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
            dependency.dstAccessMask |=
                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        }

        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;

        if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
            throw std::runtime_error("failed to create render pass!");
        }

        framebuffers.resize(numberPaths);

        for (uint32_t i = 0; i < framebuffers.size(); i++) {
            VkImageView imageView = this->getImageView(i);
            VkImageView attachments[] = {imageView,
                                         depthAttachment ? depthAttachment->getImageView(i) : VK_NULL_HANDLE};

            // TODO: create the framebuffers in a compute graph element of their own that takes
            // an ImageViewSrc.

            VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = depthAttachment ? 2u : 1u;
            framebufferInfo.pAttachments = attachments;
            framebufferInfo.width = swapChainExtent.width;
            framebufferInfo.height = swapChainExtent.height;
            framebufferInfo.layers = 1;

            if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &framebuffers[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create framebuffer!");
            }
        }

        auto cameraUBO = getCameraUBO();
        for (auto& drawComponent : drawComponents) {
            drawComponent->setNumberPaths(numberPaths);
            drawComponent->setDepthState(depthAttachment != nullptr, depthState);
            drawComponent->initialize(vulkanContext, renderPass, cameraUBO);
        }
    };

    virtual void _record(VkCommandBuffer commandBuffer, uint32_t pathId) {
        // auto& camera = vulkanContext->getCamera();
        // auto& vertices = getVertices(type);

        auto& image = getImage(pathId);

        VkImageMemoryBarrier barrierBack = {};
        barrierBack.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        // Keep the contents when drawing over them; else they are discarded.
        auto source = std::dynamic_pointer_cast<ImageViewSrc>(getInputElement(0));
        barrierBack.oldLayout = loadExisting ? source->getOutputLayout() : VK_IMAGE_LAYOUT_UNDEFINED;
        barrierBack.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrierBack.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrierBack.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrierBack.image = image;
        barrierBack.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrierBack.subresourceRange.baseMipLevel = 0;
        barrierBack.subresourceRange.levelCount = 1;
        barrierBack.subresourceRange.baseArrayLayer = 0;
        barrierBack.subresourceRange.layerCount = 1;

        barrierBack.srcAccessMask = loadExisting ? VK_ACCESS_MEMORY_WRITE_BIT : VK_ACCESS_SHADER_WRITE_BIT;
        barrierBack.dstAccessMask = loadExisting
                                        ? VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                                        : VK_ACCESS_MEMORY_READ_BIT;

        vkCmdPipelineBarrier(
            commandBuffer, loadExisting ? VK_PIPELINE_STAGE_ALL_COMMANDS_BIT : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            loadExisting ? VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT : VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0,
            nullptr, 0, nullptr, 1, &barrierBack);

        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = framebuffers[pathId];

        renderPassInfo.renderArea.offset = {0, 0};
        renderPassInfo.renderArea.extent = swapChainExtent;

        VkClearValue clearColor = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
        renderPassInfo.clearValueCount = 1;
        renderPassInfo.pClearValues = &clearColor;

        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
        for (auto& drawComponent : drawComponents) {
            drawComponent->recordCommandBuffer(commandBuffer, framebuffers[pathId], pathId);
        }

        vkCmdEndRenderPass(commandBuffer);
    };

    /**
     * @brief Draw over the image's contents instead of clearing it first.
     *
     * The image must be in VK_IMAGE_LAYOUT_GENERAL when the pass starts, as
     * ImageViewSrc leaves it. Call before the pass is compiled.
     */
    void setLoadExisting(bool load) { loadExisting = load; }

    void setDepthAttachment(std::shared_ptr<DepthAttachment> attachment, DepthState state = {}) {
        if (vulkanContext)
            throw std::logic_error("RenderPass: set depth before graph compilation");
        computeDependencies.erase(std::remove(computeDependencies.begin(), computeDependencies.end(), depthAttachment),
                                  computeDependencies.end());
        depthAttachment = std::move(attachment);
        depthState = state;
        if (depthAttachment)
            addComputeDependency(depthAttachment);
    }
    std::shared_ptr<DepthAttachment> getDepthAttachment() const { return depthAttachment; }
    VkImageLayout getOutputLayout() const override { return outputLayout; }
    std::optional<VkImageLayout> getFinalLayoutOverride() const override {
        if (inputs.empty())
            return std::nullopt;
        auto source =
            std::dynamic_pointer_cast<ImageViewSrcInterface>(const_cast<RenderPass*>(this)->getInputElement(0));
        return source ? source->getFinalLayoutOverride() : std::nullopt;
    }

    void addDrawComponent(std::shared_ptr<DrawComponent> drawComponent) { drawComponents.push_back(drawComponent); }

    // Makes `element` a scheduling dependency of this render pass: Kahn
    // ordering (and the per-edge semaphore the graph creates for it) then
    // guarantees `element`'s command buffer is submitted — and signals
    // completion — before this render pass's. This is how compute work that
    // a DrawComponent consumes (e.g. a sort/cull sub-graph feeding an
    // indirect draw, bridged through a BufferToGraphicsBarrier) gets ordered
    // ahead of the render pass, alongside the existing ImageViewSrc/camera
    // inputs at indices 0/1.
    void addComputeDependency(ComputeGraphElementPtr element) { computeDependencies.push_back(element); }

    virtual std::map<int, ComputeGraphElementPtr> getInputs() const override {
        auto allInputs = inputs;
        int index = (int)inputs.size();
        for (auto& dependency : computeDependencies) {
            allInputs[index++] = dependency;
        }
        return allInputs;
    }

    VkImageView& getImageView(uint32_t pathId) override {
        if (inputs.size() == 0) {
            throw std::runtime_error("no input!");
        }
        ImageViewSrc* imageViewSrc = std::dynamic_pointer_cast<ImageViewSrc>(getInputElement(0)).get();
        if (imageViewSrc == nullptr) {
            throw std::runtime_error("input is not an ImageViewSrc!");
        }
        return imageViewSrc->getImageView(pathId);
    }

    VkImage& getImage(uint32_t pathId) override {
        if (inputs.size() == 0) {
            throw std::runtime_error("no input!");
        }
        ImageViewSrc* imageViewSrc = std::dynamic_pointer_cast<ImageViewSrc>((getInputElement(0))).get();
        if (imageViewSrc == nullptr) {
            throw std::runtime_error("input is not an ImageViewSrc!");
        }
        return imageViewSrc->getImage(pathId);
    }

    VkExtent2D& getImageExtent(uint32_t pathId) override {
        if (inputs.size() == 0) {
            throw std::runtime_error("no input!");
        }
        ImageViewSrc* imageViewSrc = std::dynamic_pointer_cast<ImageViewSrc>((getInputElement(0))).get();
        if (imageViewSrc == nullptr) {
            throw std::runtime_error("input is not an ImageViewSrc!");
        }
        return imageViewSrc->getImageExtent(pathId);
    }

private:
    VulkanContext* vulkanContext = nullptr;
    VkRenderPass renderPass = VK_NULL_HANDLE;
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;

    std::vector<VkFramebuffer> framebuffers;

    std::vector<std::shared_ptr<DrawComponent>> drawComponents;

    std::vector<ComputeGraphElementPtr> computeDependencies;

    bool loadExisting = false;
    std::shared_ptr<DepthAttachment> depthAttachment;
    DepthState depthState{};
    VkImageLayout outputLayout = VK_IMAGE_LAYOUT_GENERAL;
};

typedef std::shared_ptr<RenderPass> RenderPassPtr;

} // namespace klartraum

#endif // KLARTRAUM_COMPUTEGRAPH_RENDERPASS_HPP
