#ifndef KLARTRAUM_CORE_HPP
#define KLARTRAUM_CORE_HPP

#include <vector>
#include <queue>
#include <optional>
#include <memory>


#include "klartraum/vulkan_kernel.hpp"
#include "klartraum/backend_config.hpp"
#include "klartraum/draw_component.hpp"
#include "klartraum/camera.hpp"
#include "klartraum/interface_camera.hpp"
#include "klartraum/events.hpp"

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/renderpass.hpp"


namespace klartraum {

class KlartraumCore {
public:
    KlartraumCore();
    ~KlartraumCore();

    void step();

    std::queue<std::unique_ptr<Event> >& getEventQueue();

    void setInterfaceCamera(std::shared_ptr<InterfaceCamera> camera);
    void setCameraUBO(std::shared_ptr<CameraUboType> cameraUBO) {
        this->cameraUBO = cameraUBO;
    }

    VulkanKernel& getVulkanKernel();

    void add(ComputeGraphElementPtr element);

    RenderPassPtr createRenderPass();

    void clearComputeGraphs() {
        computeGraphs.clear();
    }

private:
    VulkanKernel vulkanKernel;

    std::shared_ptr<InterfaceCamera> interfaceCamera;
    std::shared_ptr<CameraUboType> cameraUBO;

    std::queue<std::unique_ptr<Event> > eventQueue;

    std::vector<ComputeGraph> computeGraphs;

};

} // namespace klartraum

#endif // KLARTRAUM_CORE_HPP