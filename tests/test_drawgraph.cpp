#include <gtest/gtest.h>

#include <map>
#include <vector>

#include "klartraum/glfw_frontend.hpp"

#include "klartraum/drawgraph/drawgraph.hpp"

#include "klartraum/drawgraph/imageviewsrc.hpp"
#include "klartraum/drawgraph/renderpass.hpp"
#include "klartraum/draw_basics.hpp"

using namespace klartraum;

class BlurOp : public DrawGraphElement {
    virtual const char* getName() const {
        return "BlurOp";
    }

    virtual void checkInput(DrawGraphElementPtr input, int index = 0) {
        // accept everything
    }

    virtual void _record(VkCommandBuffer commandBuffer) {};

};

class NoiseOp : public DrawGraphElement {
    virtual const char* getName() const {
        return "NoiseOp";
    }

    virtual void checkInput(DrawGraphElementPtr input, int index = 0) {
        // accept everything
    }

    virtual void _record(VkCommandBuffer commandBuffer) {};

};

class AddOp : public DrawGraphElement {
    virtual const char* getName() const {
        return "AddOp";
    }

    virtual void checkInput(DrawGraphElementPtr input, int index = 0) {
        // accept everything
    }

    virtual void _record(VkCommandBuffer commandBuffer) {};

};

class CopyOp : public DrawGraphElement {
    virtual const char* getName() const {
        return "CopyOp";
    }

    virtual void checkInput(DrawGraphElementPtr input, int index = 0) {
        // accept everything
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
    std::vector<VkImageView> imageViews;
    std::vector<VkSemaphore> imageAvailableSemaphores;
    for (int i = 0; i < 3; i++) {
        imageViews.push_back(vulkanKernel.getImageView(i));
    }

    auto imageViewSrc = std::make_shared<ImageViewSrc>(imageViews);

    auto swapChainImageFormat = vulkanKernel.getSwapChainImageFormat();
    auto swapChainExtent = vulkanKernel.getSwapChainExtent();
    auto renderpass = std::make_shared<RenderPass>(swapChainImageFormat, swapChainExtent);

    renderpass->setInput(imageViewSrc);

    auto blur = std::make_shared<BlurOp>();
    blur->setInput(renderpass);
    auto noise = std::make_shared<NoiseOp>();
    noise->setInput(blur);

    auto add = std::make_shared<AddOp>();
    add->setInput(blur, 0);
    add->setInput(noise, 1);

    auto copy = std::make_shared<CopyOp>();
    copy->setInput(add);

    
    /*
    STEP 2: create the drawgraph backend
    */
    
    // this traverses the drawgraph and creates the vulkan objects
    auto& drawgraph = DrawGraph(vulkanKernel, 1);
    drawgraph.compileFrom(copy);

    drawgraph.submitAndWait(vulkanKernel.getGraphicsQueue(), 0);

    return;
}

TEST(DrawGraph, trippleFramebuffer) {
    klartraum::GlfwFrontend frontend;

    auto& core = frontend.getKlartraumCore();
    auto& vulkanKernel = core.getVulkanKernel();
    auto& device = vulkanKernel.getDevice();

    /*
    STEP 1: create the drawgraph elements
    */

    std::vector<VkImageView> imageViews;
    std::vector<VkSemaphore> imageAvailableSemaphores;

    for (int i = 0; i < 3; i++) {
        imageViews.push_back(vulkanKernel.getImageView(i));
        imageAvailableSemaphores.push_back(vulkanKernel.imageAvailableSemaphoresPerImage[i]);
    }

    auto imageViewSrc = std::make_shared<ImageViewSrc>(imageViews);
    imageViewSrc->setWaitFor(0, imageAvailableSemaphores[0]);
    imageViewSrc->setWaitFor(1, imageAvailableSemaphores[1]);
    imageViewSrc->setWaitFor(2, imageAvailableSemaphores[2]);

    auto camera = std::make_shared<CameraUboType>();

    auto swapChainImageFormat = vulkanKernel.getSwapChainImageFormat();
    auto swapChainExtent = vulkanKernel.getSwapChainExtent();
    auto renderpass = std::make_shared<RenderPass>(swapChainImageFormat, swapChainExtent);

    renderpass->setInput(imageViewSrc, 0);
    renderpass->setInput(camera, 1);

    auto axes = std::make_shared<DrawBasics>(DrawBasicsType::Axes);
    renderpass->addDrawComponent(axes);

    /*
    STEP 2: create the drawgraph backend
    */
    
    // this traverses the drawgraph and creates the vulkan objects
    auto& drawgraph = DrawGraph(vulkanKernel, 3);

    drawgraph.compileFrom(renderpass);

    VkSemaphore finishSemaphore = VK_NULL_HANDLE;

    for(int i = 0; i < 6; i++) {
        auto [imageIndex, imageAvailableSemaphore] = vulkanKernel.beginRender();
        finishSemaphore = drawgraph.submitTo(vulkanKernel.getGraphicsQueue(), imageIndex);
        vulkanKernel.endRender(imageIndex, finishSemaphore);
    }

    // finally wait for the queue to be completed
    // by submitting a fence without any work
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence;
    if (vkCreateFence(device, &fenceInfo, nullptr, &fence) != VK_SUCCESS) {
        throw std::runtime_error("failed to create fence!");
    }

    if (vkQueueSubmit(vulkanKernel.getGraphicsQueue(), 0, nullptr, fence) != VK_SUCCESS) {
        throw std::runtime_error("failed to submit the fence!");
    }

    vkWaitForFences(device, 1, &fence, true, UINT64_MAX);

    vkDestroyFence(device, fence, nullptr);  
    return;
}