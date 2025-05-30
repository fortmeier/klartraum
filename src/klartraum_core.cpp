#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "klartraum/klartraum_core.hpp"

#include "klartraum/drawgraph/drawgraph.hpp"
#include "klartraum/drawgraph/imageviewsrc.hpp"
#include "klartraum/drawgraph/renderpass.hpp"

namespace klartraum
{

KlartraumCore::KlartraumCore() {

}

KlartraumCore::~KlartraumCore() {

}

void KlartraumCore::step() {

    // start frame rendering
    auto [imageIndex, semaphore] = vulkanKernel.beginRender();

    // process event queue,
    // this currently only updates the camera
    if(interfaceCamera != nullptr)
    {
        while (!eventQueue.empty()) {
            auto event = std::move(eventQueue.front());
            eventQueue.pop();
            interfaceCamera->onEvent(*event);
        }
        auto& camera = vulkanKernel.getCamera();
        interfaceCamera->update(camera);
    }

    auto& graphicsQueue = vulkanKernel.getGraphicsQueue();

    VkSemaphore renderFinishedSemaphore;
    for(auto &drawGraph : drawGraphs) {
        renderFinishedSemaphore = drawGraph.submitTo(graphicsQueue, imageIndex);
    }

    // finish frame rendering
    vulkanKernel.endRender(imageIndex, renderFinishedSemaphore);
 
}

std::queue<std::unique_ptr<Event> >& KlartraumCore::getEventQueue()
{
    return eventQueue;
}


void KlartraumCore::setInterfaceCamera(std::shared_ptr<InterfaceCamera> camera)
{
    camera->initialize(vulkanKernel);
    this->interfaceCamera = camera;
}

VulkanKernel& KlartraumCore::getVulkanKernel()
{
    return vulkanKernel;
}

void KlartraumCore::add(DrawGraphElementPtr element)
{
    drawGraphs.emplace_back(vulkanKernel, 3);
    auto& drawGraph = drawGraphs.back();
    drawGraph.compileFrom(element);
}

RenderPassPtr KlartraumCore::createRenderPass()
{
    std::vector<VkImageView> imageViews;
    std::vector<VkImage> images;
    std::vector<VkSemaphore> imageAvailableSemaphores;

    for (int i = 0; i < 3; i++) {
        imageViews.push_back(vulkanKernel.getImageView(i));
        images.push_back(vulkanKernel.getSwapChainImage(i));
        imageAvailableSemaphores.push_back(vulkanKernel.imageAvailableSemaphoresPerImage[i]);
    }

    auto imageViewSrc = std::make_shared<ImageViewSrc>(imageViews, images);

    imageViewSrc->setWaitFor(0, imageAvailableSemaphores[0]);
    imageViewSrc->setWaitFor(1, imageAvailableSemaphores[1]);
    imageViewSrc->setWaitFor(2, imageAvailableSemaphores[2]);    

    auto swapChainImageFormat = vulkanKernel.getSwapChainImageFormat();
    auto swapChainExtent = vulkanKernel.getSwapChainExtent();
    auto renderpass = std::make_shared<RenderPass>(swapChainImageFormat, swapChainExtent);

    renderpass->setInput(imageViewSrc);
    return renderpass;
}

} // namespace klartraum