#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "klartraum/klartraum_core.hpp"

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/imageviewsrc.hpp"
#include "klartraum/computegraph/renderpass.hpp"

namespace klartraum
{

KlartraumEngine::KlartraumEngine() {

}

KlartraumEngine::~KlartraumEngine() {

}

void KlartraumEngine::step() {

    // start frame rendering
    auto [imageIndex, fence] = vulkanContext.beginRender();

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

    for(auto it = computeGraphs.begin(); it != computeGraphs.end(); ++it) {
        // only the last computegraph submits the fence and returns the renderFinishedSemaphore
        // should be done somewhat different
        if (it == computeGraphs.end() - 1) {
            renderFinishedSemaphore = (*it)->submitTo(graphicsQueue, imageIndex, fence);
        }
        else {
            (*it)->submitTo(graphicsQueue, imageIndex);
        }
    }

    // finish frame rendering
    vulkanContext.endRender(imageIndex, renderFinishedSemaphore);

    // When profiling is enabled, wait for the GPU to finish this frame so we
    // can read back the timestamp queries.  This makes each step() synchronous
    // but that is acceptable in a profiling / development build.
    if (profilingEnabled_) {
        vkQueueWaitIdle(graphicsQueue);
        for (auto& cg : computeGraphs) {
            cg->readAndAccumulateTimestamps_();
        }
    }
}

std::queue<std::unique_ptr<Event> >& KlartraumEngine::getEventQueue()
{
    return eventQueue;
}


void KlartraumEngine::setInterfaceCamera(std::shared_ptr<InterfaceCamera> camera)
{
    camera->initialize(vulkanContext);
    this->interfaceCamera = camera;
}

VulkanContext& KlartraumEngine::getVulkanContext()
{
    return vulkanContext;
}

void KlartraumEngine::add(ComputeGraphElementPtr element)
{
    uint32_t numberPaths = vulkanContext.getNumberOfSwapChainImages();
    computeGraphs.emplace_back(std::make_unique<ComputeGraph>(vulkanContext, numberPaths));
    auto& computeGraph = computeGraphs.back();
    if (profilingEnabled_) computeGraph->enableProfiling();
    if (perfProfilingEnabled_) computeGraph->enablePerformanceProfiling(perfProfilingNameFilter_);
    computeGraph->compileFrom(element);
}

std::vector<std::pair<std::string, float>> KlartraumEngine::getProfilingResults()
{
    vkQueueWaitIdle(vulkanContext.getGraphicsQueue());
    for (auto& cg : computeGraphs) {
        cg->readAndAccumulateTimestamps_();
        cg->readAndAccumulatePerformanceCounters_();
    }
    std::vector<std::pair<std::string, float>> results;
    for (auto& cg : computeGraphs) {
        auto r = cg->getProfilingResults();
        results.insert(results.end(), r.begin(), r.end());
    }
    return results;
}

RenderPassPtr KlartraumEngine::createRenderPass()
{
    std::vector<VkImageView> imageViews;
    std::vector<VkImage> images;
    std::vector<VkExtent2D> imageExtents;
    std::vector<VkSemaphore> imageAvailableSemaphores;

    for (int i = 0; i < vulkanContext.getNumberOfSwapChainImages(); i++) {
        imageViews.push_back(vulkanContext.getImageView(i));
        images.push_back(vulkanContext.getSwapChainImage(i));
        imageExtents.push_back(vulkanContext.getSwapChainExtent());
        imageAvailableSemaphores.push_back(vulkanContext.imageAvailableSemaphoresPerImage[i]);
    }

    auto imageViewSrc = std::make_shared<ImageViewSrc>(imageViews, images, imageExtents);

    for (int i = 0; i < vulkanContext.getNumberOfSwapChainImages(); i++) {
        imageViewSrc->setWaitFor(i, imageAvailableSemaphores[i]);
    }

    auto camera = std::make_shared<CameraUboType>();

    auto swapChainImageFormat = vulkanContext.getSwapChainImageFormat();
    auto swapChainExtent = vulkanContext.getSwapChainExtent();
    auto renderpass = std::make_shared<RenderPass>(swapChainImageFormat, swapChainExtent);

    renderpass->setInput(imageViewSrc, 0);
    renderpass->setInput(camera, 1);

    return renderpass;
}

} // namespace klartraum