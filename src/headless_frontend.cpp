#include <stdexcept>

#include "klartraum/headless_frontend.hpp"
//##include "klartraum/events.hpp"

namespace klartraum {


HeadlessFrontend::HeadlessFrontend()
{
    klartraumEngine = std::make_unique<KlartraumEngine>();

    initialize();
}

HeadlessFrontend::~HeadlessFrontend()
{
    shutdown();
}

void HeadlessFrontend::initialize() {
    klartraumEngine->getVulkanContext().initialize();
}


// void HeadlessFrontend::loop() {


// }

void HeadlessFrontend::shutdown() {
    auto& instance = klartraumEngine->getVulkanContext().getInstance();
    auto& vulkanContext = klartraumEngine->getVulkanContext();
    
    vulkanContext.stopRender();
    // The graph builder may hold GPU resources (e.g. a captured model).
    klartraumEngine->setGraphBuilder(nullptr);
    klartraumEngine->clearComputeGraphs();
    // Engine-held buffers (camera UBO, interface camera) must be released
    // while the device is still valid.
    klartraumEngine->setCameraUBO(nullptr);
    klartraumEngine->clearInterfaceCamera();
    klartraumEngine->clearWindow();
    vulkanContext.shutdown();

    //vkDestroySurfaceKHR(instance, surface, nullptr);
}

KlartraumEngine& HeadlessFrontend::getKlartraumEngine()
{
    return *klartraumEngine;
}

} // namespace klartraum