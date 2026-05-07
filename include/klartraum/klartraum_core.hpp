#ifndef KLARTRAUM_CORE_HPP
#define KLARTRAUM_CORE_HPP

#include <string>
#include <utility>
#include <vector>
#include <queue>
#include <optional>
#include <memory>


#include "klartraum/vulkan_context.hpp"
#include "klartraum/backend_config.hpp"
#include "klartraum/draw_component.hpp"
#include "klartraum/camera.hpp"
#include "klartraum/interface_camera.hpp"
#include "klartraum/events.hpp"

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/renderpass.hpp"


namespace klartraum {

class KlartraumEngine {
public:
    KlartraumEngine();
    ~KlartraumEngine();

    void step();

    std::queue<std::unique_ptr<Event> >& getEventQueue();

    void setInterfaceCamera(std::shared_ptr<InterfaceCamera> camera);
    void setCameraUBO(std::shared_ptr<CameraUboType> cameraUBO) {
        this->cameraUBO = cameraUBO;
    }

    VulkanContext& getVulkanContext();

    void add(ComputeGraphElementPtr element);

    RenderPassPtr createRenderPass();

    void clearComputeGraphs() {
        computeGraphs.clear();
    }

    void clearInterfaceCamera() {
        interfaceCamera = nullptr;
    }

    // Call before add() to enable GPU timestamp profiling on all subsequent
    // compute graphs.  Results accumulate across frames and are averaged.
    void enableProfiling() { profilingEnabled_ = true; }

    // Waits for the GPU to be idle, reads the last frame's timestamps, and
    // returns {elementName, meanTimeMs} for every element across all graphs.
    std::vector<std::pair<std::string, float>> getProfilingResults();

private:
    bool profilingEnabled_ = false;

    VulkanContext vulkanContext;

    std::shared_ptr<InterfaceCamera> interfaceCamera;
    std::shared_ptr<CameraUboType> cameraUBO;

    std::queue<std::unique_ptr<Event> > eventQueue;

    std::vector<std::unique_ptr<ComputeGraph>> computeGraphs;

};

} // namespace klartraum

#endif // KLARTRAUM_CORE_HPP