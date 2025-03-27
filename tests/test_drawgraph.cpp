#include <gtest/gtest.h>

#include <map>
#include <vector>

#include "klartraum/glfw_frontend.hpp"

#include "klartraum/drawgraph/drawgraph.hpp"

using namespace klartraum;

class FramebufferSrc : public DrawGraphElement {
public:
    FramebufferSrc(VkFramebuffer framebuffer) : framebuffer(framebuffer) {
    }; 

    virtual const char* getName() const {
        return "FramebufferSrc";
    }

    //uint32_t framebuffer_index;
    VkFramebuffer framebuffer;

    virtual void _record(VkCommandBuffer commandBuffer) {

    };

};

class RenderPass : public DrawGraphElement {
public:
    RenderPass(VkFormat swapChainImageFormat, VkExtent2D extent) :
        swapChainImageFormat(swapChainImageFormat),
        swapChainExtent(extent) {
    
    };

    virtual const char* getName() const {
        return "RenderPass";
    }

    virtual void _setup(VkDevice& device) {
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
    };


    virtual void _record(VkCommandBuffer commandBuffer) {
        // auto& camera = vulkanKernel->getCamera();
        // auto& vertices = getVertices(type);

        
        FramebufferSrc* framebufferSrc = std::dynamic_pointer_cast<FramebufferSrc>(inputs[0]).get();
        if (!framebufferSrc) {
            throw std::runtime_error("FramebufferSrc not set!");
        }

        VkFramebuffer framebuffer = framebufferSrc->framebuffer;
        
        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = framebuffer;
    
        renderPassInfo.renderArea.offset = { 0, 0 };
        renderPassInfo.renderArea.extent = swapChainExtent;
    
        VkClearValue clearColor = { {{0.0f, 0.0f, 0.0f, 1.0f}} };
        renderPassInfo.clearValueCount = 1;
        renderPassInfo.pClearValues = &clearColor;
    
        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);   
    
        // vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
    
        // VkViewport viewport{};
        // viewport.x = 0.0f;
        // viewport.y = 0.0f;
        // viewport.width = static_cast<float>(swapChainExtent.width);
        // viewport.height = static_cast<float>(swapChainExtent.height);
        // viewport.minDepth = 0.0f;
        // viewport.maxDepth = 1.0f;
        // vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
    
        // VkRect2D scissor{};
        // scissor.offset = { 0, 0 };
        // scissor.extent = swapChainExtent;
        // vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
    
        // VkBuffer vertexBuffers[] = {vertexBuffer};
        // VkDeviceSize offsets[] = {0};
        // vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
        // auto& descriptorSets = camera.getDescriptorSets();
        // vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);
        // vkCmdDraw(commandBuffer, static_cast<uint32_t>(vertices.size()), 1, 0, 0);
    
        vkCmdEndRenderPass(commandBuffer);

    };

private:
    VkRenderPass renderPass;
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;

};

class BlurOp : public DrawGraphElement {
    virtual const char* getName() const {
        return "BlurOp";
    }

    virtual void _record(VkCommandBuffer commandBuffer) {};

};

class NoiseOp : public DrawGraphElement {
    virtual const char* getName() const {
        return "NoiseOp";
    }

    virtual void _record(VkCommandBuffer commandBuffer) {};

};

class AddOp : public DrawGraphElement {
    virtual const char* getName() const {
        return "AddOp";
    }

    virtual void _record(VkCommandBuffer commandBuffer) {};

};

class CopyOp : public DrawGraphElement {
    virtual const char* getName() const {
        return "CopyOp";
    }

    virtual void _record(VkCommandBuffer commandBuffer) {};

};

class ImageViewSrc : public DrawGraphElement {
    virtual const char* getName() const {
        return "ImageViewSrc";
    }

    virtual void _record(VkCommandBuffer commandBuffer) {};
};




TEST(DrawGraph, create) {
    klartraum::GlfwFrontend frontend;

    auto& core = frontend.getKlartraumCore();
    auto& vulkanKernel = core.getVulkanKernel();
    auto& device = vulkanKernel.getDevice();

    /*
    STEP 1: create the drawgraph elements
    */

    auto framebuffer_src = std::make_shared<FramebufferSrc>(vulkanKernel.getFramebuffer(0));

    auto swapChainImageFormat = vulkanKernel.getSwapChainImageFormat();
    auto swapChainExtent = vulkanKernel.getSwapChainExtent();
    auto renderpass = std::make_shared<RenderPass>(swapChainImageFormat, swapChainExtent);

    renderpass->set_input(framebuffer_src);

    auto blur = std::make_shared<BlurOp>();
    blur->set_input(renderpass);
    auto noise = std::make_shared<NoiseOp>();
    noise->set_input(blur);

    auto add = std::make_shared<AddOp>();
    add->set_input(blur, 0);
    add->set_input(noise, 1);

    auto copy = std::make_shared<CopyOp>();
    copy->set_input(add);

    
    /*
    STEP 2: create the drawgraph backend
    */
    
    // this traverses the drawgraph and creates the vulkan objects
    auto& drawgraph = DrawGraph(vulkanKernel);
    drawgraph.compileFrom(copy);

    drawgraph.submitTo(vulkanKernel.getGraphicsQueue());

    return;
}