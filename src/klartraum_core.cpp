#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "klartraum/klartraum_core.hpp"

namespace klartraum
{

/* 
void KlartraumCore::initialize()
{
    camera = std::make_unique<Camera>(this);
}

void KlartraumCore::shutdown() {
    camera = nullptr;    
}
*/

void KlartraumCore::step() {

    uint32_t imageIndex = vulkanKernel.beginRender();


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


    auto& currentFrame = vulkanKernel.currentFrame;
    auto& framebuffer = vulkanKernel.getFramebuffer(currentFrame);
    auto& commandBuffer = vulkanKernel.commandBuffers[currentFrame];
    auto& imageAvailableSemaphore = vulkanKernel.imageAvailableSemaphores[currentFrame];

    for(auto &drawComponent : drawComponents) {
        drawComponent->draw(currentFrame, commandBuffer, framebuffer, imageAvailableSemaphore);
    }
    vulkanKernel.endRender(imageIndex);
 
}

std::queue<std::unique_ptr<Event> >& KlartraumCore::getEventQueue()
{
    return eventQueue;
}


void KlartraumCore::setInterfaceCamera(std::shared_ptr<InterfaceCamera> camera)
{
    this->interfaceCamera = camera;
}

void KlartraumCore::addDrawComponent(std::unique_ptr<DrawComponent> drawComponent)
{
    drawComponents.push_back(std::move(drawComponent));
}

VulkanKernel& KlartraumCore::getVulkanKernel()
{
    return vulkanKernel;
}

} // namespace klartraum