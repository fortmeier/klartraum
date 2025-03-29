#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "klartraum/klartraum_core.hpp"

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

    // draw all draw components
    // auto& currentFrame = vulkanKernel.currentFrame;
    // auto& framebuffer = vulkanKernel.getFramebuffer(imageIndex);
    // auto& commandBuffer = vulkanKernel.commandBuffers[currentFrame];
    // auto& imageAvailableSemaphore = vulkanKernel.imageAvailableSemaphores[currentFrame];

    // for(auto &drawComponent : drawComponents) {
    //     drawComponent->draw(currentFrame, commandBuffer, framebuffer, imageAvailableSemaphore, imageIndex);
    // }
    auto& graphicsQueue = vulkanKernel.getGraphicsQueue();

    VkSemaphore renderFinishedSemaphore;
    for(auto &drawGraph : drawGraphs) {
        renderFinishedSemaphore = drawGraph.submitToWithSemaphore(graphicsQueue, imageIndex);
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

} // namespace klartraum