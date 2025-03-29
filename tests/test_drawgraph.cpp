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
    auto& drawgraph = DrawGraph(vulkanKernel, 1);
    drawgraph.compileFrom(copy);

    drawgraph.submitTo(vulkanKernel.getGraphicsQueue(), 0);

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
    for (int i = 0; i < 3; i++) {
        imageViews.push_back(vulkanKernel.getImageView(i));
    }
    auto imageViewSrc = std::make_shared<ImageViewSrc>(imageViews);

    auto swapChainImageFormat = vulkanKernel.getSwapChainImageFormat();
    auto swapChainExtent = vulkanKernel.getSwapChainExtent();
    auto renderpass = std::make_shared<RenderPass>(swapChainImageFormat, swapChainExtent);

    renderpass->set_input(imageViewSrc);

    auto axes = std::make_shared<DrawBasics>(DrawBasicsType::Axes);
    renderpass->addDrawComponent(axes);

    /*
    STEP 2: create the drawgraph backend
    */
    
    // this traverses the drawgraph and creates the vulkan objects
    auto& drawgraph = DrawGraph(vulkanKernel, 3);
    drawgraph.compileFrom(renderpass);

    auto [imageIndex, semaphore] = vulkanKernel.beginRender();
    auto finishSemaphore = drawgraph.submitToWithSemaphore(vulkanKernel.getGraphicsQueue(), imageIndex);
    vulkanKernel.endRender(imageIndex, finishSemaphore);

    return;
}