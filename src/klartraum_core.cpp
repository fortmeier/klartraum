#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "klartraum/klartraum_core.hpp"

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/imageviewsrc.hpp"
#include "klartraum/computegraph/renderpass.hpp"

namespace klartraum
{

KlartraumCore::KlartraumCore() {

}

KlartraumCore::~KlartraumCore() {

}

void KlartraumCore::step() {

    // start frame rendering
    auto [imageIndex, semaphore] = vulkanContext.beginRender();

    // process event queue,
    // this currently only updates the camera
    if(interfaceCamera != nullptr && cameraUBO != nullptr)
    {
        while (!eventQueue.empty()) {
            auto event = std::move(eventQueue.front());
            eventQueue.pop();
            interfaceCamera->onEvent(*event);
        }
        
        interfaceCamera->update(cameraUBO->ubo);
        cameraUBO->update(imageIndex);
    }

    auto& graphicsQueue = vulkanContext.getGraphicsQueue();

    VkSemaphore renderFinishedSemaphore;
    for(auto &computeGraph : computeGraphs) {
        renderFinishedSemaphore = computeGraph.submitTo(graphicsQueue, imageIndex);
    }

    // finish frame rendering
    vulkanContext.endRender(imageIndex, renderFinishedSemaphore);
 
}

std::queue<std::unique_ptr<Event> >& KlartraumCore::getEventQueue()
{
    return eventQueue;
}


void KlartraumCore::setInterfaceCamera(std::shared_ptr<InterfaceCamera> camera)
{
    camera->initialize(vulkanContext);
    this->interfaceCamera = camera;
}

VulkanContext& KlartraumCore::getVulkanContext()
{
    return vulkanContext;
}

void KlartraumCore::add(ComputeGraphElementPtr element)
{
    computeGraphs.emplace_back(vulkanContext, 3);
    auto& computeGraph = computeGraphs.back();
    computeGraph.compileFrom(element);
}

RenderPassPtr KlartraumCore::createRenderPass()
{
    std::vector<VkImageView> imageViews;
    std::vector<VkImage> images;
    std::vector<VkSemaphore> imageAvailableSemaphores;

    for (int i = 0; i < 3; i++) {
        imageViews.push_back(vulkanContext.getImageView(i));
        images.push_back(vulkanContext.getSwapChainImage(i));
        imageAvailableSemaphores.push_back(vulkanContext.imageAvailableSemaphoresPerImage[i]);
    }

    auto imageViewSrc = std::make_shared<ImageViewSrc>(imageViews, images);

    imageViewSrc->setWaitFor(0, imageAvailableSemaphores[0]);
    imageViewSrc->setWaitFor(1, imageAvailableSemaphores[1]);
    imageViewSrc->setWaitFor(2, imageAvailableSemaphores[2]);

    auto camera = std::make_shared<CameraUboType>();

    auto swapChainImageFormat = vulkanContext.getSwapChainImageFormat();
    auto swapChainExtent = vulkanContext.getSwapChainExtent();
    auto renderpass = std::make_shared<RenderPass>(swapChainImageFormat, swapChainExtent);

    renderpass->setInput(imageViewSrc, 0);
    renderpass->setInput(camera, 1);

    return renderpass;
}

} // namespace klartraum