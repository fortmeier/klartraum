#ifndef KLARTRAUM_COMPUTEGRAPH_RENDERPASS_HPP
#define KLARTRAUM_COMPUTEGRAPH_RENDERPASS_HPP

#include <map>
#include <vector>
#include <vulkan/vulkan.h>
#include "klartraum/computegraph/imageviewsrc.hpp"
#include "klartraum/computegraph/rendergraphelement.hpp"

namespace klartraum {

class RenderPass : public ImageViewSrc, public RenderGraphElement {
public:
    RenderPass(VkFormat swapChainImageFormat, VkExtent2D extent) :
        swapChainImageFormat(swapChainImageFormat),
        swapChainExtent(extent) {
    
    };

    ~RenderPass() {
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

    virtual const char* getType() const {
        return "RenderPass";
    }

    virtual void _setup(VulkanContext& vulkanContext, uint32_t numberPaths) {
        this->vulkanContext = &vulkanContext;
        
        auto& device = vulkanContext.getDevice();

        VkAttachmentDescription colorAttachment{};
        colorAttachment.format = swapChainImageFormat;
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_GENERAL;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    
        VkAttachmentReference colorAttachmentRef{};
        colorAttachmentRef.attachment = 0;
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_GENERAL;
    
        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;
    
        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = 1;
        renderPassInfo.pAttachments = &colorAttachment;
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
    
    
    
        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;
    
        if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
            throw std::runtime_error("failed to create render pass!");
        }

        framebuffers.resize(numberPaths);

        for (uint32_t i = 0; i < framebuffers.size(); i++) {
            VkImageView imageView = this->getImageView(i);
            VkImageView attachments[] = {
                imageView
            }; 
    
            // TODO create framebuffer with render pass computegraph element
            // and create imageviewsrc instead of framebuffer src
    
            VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = 1;
            framebufferInfo.pAttachments = attachments;
            framebufferInfo.width = swapChainExtent.width;
            framebufferInfo.height = swapChainExtent.height;
            framebufferInfo.layers = 1;
    
            if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &framebuffers[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create framebuffer!");
            }
        }

        auto cameraUBO = getCameraUBO();
        for(auto& drawComponent : drawComponents) {
            drawComponent->initialize(vulkanContext, renderPass, cameraUBO);
        }
    };


    virtual void _record(VkCommandBuffer commandBuffer, uint32_t pathId) {
        // auto& camera = vulkanContext->getCamera();
        // auto& vertices = getVertices(type);

        auto& image = getImage(pathId);

        VkImageMemoryBarrier barrierBack = {};
        barrierBack.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrierBack.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrierBack.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrierBack.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrierBack.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrierBack.image = image;
        barrierBack.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrierBack.subresourceRange.baseMipLevel = 0;
        barrierBack.subresourceRange.levelCount = 1;
        barrierBack.subresourceRange.baseArrayLayer = 0;
        barrierBack.subresourceRange.layerCount = 1;
        
        barrierBack.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrierBack.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        
        vkCmdPipelineBarrier(
            commandBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            0,
            0, nullptr,
            0, nullptr,
            1, &barrierBack);        
       
        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = framebuffers[pathId];
    
        renderPassInfo.renderArea.offset = { 0, 0 };
        renderPassInfo.renderArea.extent = swapChainExtent;
    
        VkClearValue clearColor = { {{0.0f, 0.0f, 0.0f, 1.0f}} };
        renderPassInfo.clearValueCount = 1;
        renderPassInfo.pClearValues = &clearColor;
    
        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
        for(auto& drawComponent : drawComponents) {
            drawComponent->recordCommandBuffer(commandBuffer, framebuffers[pathId], pathId);
        }
    
        vkCmdEndRenderPass(commandBuffer);

    };

    void addDrawComponent(std::shared_ptr<DrawComponent> drawComponent) {
        drawComponents.push_back(drawComponent);
    }

    VkImageView& getImageView(uint32_t pathId) override {
        if(inputs.size() == 0) {
            throw std::runtime_error("no input!");
        }
        ImageViewSrc* imageViewSrc = std::dynamic_pointer_cast<ImageViewSrc>(getInputElement(0)).get();
        if (imageViewSrc == nullptr) {
            throw std::runtime_error("input is not an ImageViewSrc!");
        }
        return imageViewSrc->getImageView(pathId);
    }

    VkImage& getImage(uint32_t pathId) override {
        if(inputs.size() == 0) {
            throw std::runtime_error("no input!");
        }
        ImageViewSrc* imageViewSrc = std::dynamic_pointer_cast<ImageViewSrc>((getInputElement(0))).get();
        if (imageViewSrc == nullptr) {
            throw std::runtime_error("input is not an ImageViewSrc!");
        }
        return imageViewSrc->getImage(pathId);
    }

private:
    VulkanContext* vulkanContext = nullptr;
    VkRenderPass renderPass;
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;

    std::vector<VkFramebuffer> framebuffers;

    std::vector<std::shared_ptr<DrawComponent> > drawComponents;

};

typedef std::shared_ptr<RenderPass> RenderPassPtr;

} // namespace klartraum

#endif // KLARTRAUM_COMPUTEGRAPH_RENDERPASS_HPP