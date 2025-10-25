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
    klartraumEngine->clearComputeGraphs();
    vulkanContext.shutdown();

    //vkDestroySurfaceKHR(instance, surface, nullptr);
}

KlartraumEngine& HeadlessFrontend::getKlartraumEngine()
{
    return *klartraumEngine;
}

} // namespace klartraum