#ifndef KLARTRAUM_DRAWGRAPH_RENDERPASS_HPP
#define KLARTRAUM_DRAWGRAPH_RENDERPASS_HPP

#include <map>
#include <vector>
#include <vulkan/vulkan.h>
#include "klartraum/drawgraph/drawgraphelement.hpp"
#include "klartraum/drawgraph/imageviewsrc.hpp"

namespace klartraum {

class RenderPass : public ImageViewSrc {
public:
    RenderPass(VkFormat swapChainImageFormat, VkExtent2D extent) :
        swapChainImageFormat(swapChainImageFormat),
        swapChainExtent(extent) {
    
    };

    ~RenderPass() {
        auto& device = vulkanKernel->getDevice();
        for (auto framebuffer : framebuffers) {
            vkDestroyFramebuffer(device, framebuffer, nullptr);
        }
        vkDestroyRenderPass(device, renderPass, nullptr);

    };

    virtual void checkInput(DrawGraphElementPtr input, int index = 0) {
        ImageViewSrc* imageViewSrc = std::dynamic_pointer_cast<ImageViewSrc>(input).get();
        if (imageViewSrc == nullptr) {
            throw std::runtime_error("input is not an ImageViewSrc!");
        }
    }

    virtual const char* getName() const {
        return "RenderPass";
    }

    virtual void _setup(VulkanKernel& vulkanKernel, uint32_t numberPaths) {
        this->vulkanKernel = &vulkanKernel;
        
        auto& device = vulkanKernel.getDevice();

        VkAttachmentDescription colorAttachment{};
        colorAttachment.format = swapChainImageFormat;
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
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
    
            // TODO create framebuffer with render pass drawgraph element
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

        for(auto& drawComponent : drawComponents) {
            drawComponent->initialize(vulkanKernel, renderPass);
        }
    };


    virtual void _record(VkCommandBuffer commandBuffer, uint32_t pathId) {
        // auto& camera = vulkanKernel->getCamera();
        // auto& vertices = getVertices(type);

        
       
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
        ImageViewSrc* imageViewSrc = std::dynamic_pointer_cast<ImageViewSrc>(inputs[0]).get();
        if (imageViewSrc == nullptr) {
            throw std::runtime_error("input is not an ImageViewSrc!");
        }
        return imageViewSrc->getImageView(pathId);
    }

    VkImage& getImage(uint32_t pathId) override {
        if(inputs.size() == 0) {
            throw std::runtime_error("no input!");
        }
        ImageViewSrc* imageViewSrc = std::dynamic_pointer_cast<ImageViewSrc>(inputs[0]).get();
        if (imageViewSrc == nullptr) {
            throw std::runtime_error("input is not an ImageViewSrc!");
        }
        return imageViewSrc->getImage(pathId);
    }

private:
    VulkanKernel* vulkanKernel = nullptr;
    VkRenderPass renderPass;
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;

    std::vector<VkFramebuffer> framebuffers;

    std::vector<std::shared_ptr<DrawComponent> > drawComponents;

};

typedef std::shared_ptr<RenderPass> RenderPassPtr;

} // namespace klartraum

#endif // KLARTRAUM_DRAWGRAPH_RENDERPASS_HPP