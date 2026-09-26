#ifndef KLARTRAUM_CORE_HPP
#define KLARTRAUM_CORE_HPP

#include <functional>
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
#include "klartraum/frame_overlay.hpp"

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/renderpass.hpp"
#include "klartraum/window.hpp"


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

    // The window's viewport compositor (lazily created). makeViewport() hands out
    // offscreen render targets that step() composites into the single swapchain
    // image before presenting. See examples/multi_viewports_single_camera_example.cpp.
    Window& getWindow();

    void add(ComputeGraphElementPtr element);

    // Builds everything that depends on the swapchain: the builder creates the
    // swapchain-backed elements (ImageViewSrc, render backends, camera UBO) and
    // registers them via add()/setCameraUBO(). It runs once immediately and
    // again whenever the swapchain is recreated after a window resize, with
    // all previously added compute graphs already released. Data that should
    // survive a resize (e.g. a GaussianDataStandard) belongs outside the
    // builder, captured by it. Setting a builder makes the window resizable.
    using GraphBuilder = std::function<void(KlartraumEngine&)>;
    void setGraphBuilder(GraphBuilder builder);

    // A window can be resized when a graph builder rebuilds the graphs for the
    // new swapchain. Viewport mode (getWindow().makeViewport()) is not covered.
    bool isResizable() const;

    RenderPassPtr createRenderPass();

    void clearComputeGraphs() {
        computeGraphs.clear();
    }

    // Release the window and its viewport offscreen images. Frontends must call
    // this before VulkanContext::shutdown() so the images are freed while the
    // device is still valid.
    void clearWindow() {
        window_.reset();
    }

    void clearInterfaceCamera() {
        interfaceCamera = nullptr;
    }

    // Registers per-frame work drawn on top of each finished frame (e.g. a
    // GUI); see FrameOverlay. Pass nullptr to remove it. Frontends must remove
    // the overlay before VulkanContext::shutdown() so its GPU resources are
    // freed while the device is still valid.
    void setOverlay(std::shared_ptr<FrameOverlay> overlay) {
        overlay_ = std::move(overlay);
    }

    // Call before add() to enable GPU timestamp profiling on all subsequent
    // compute graphs.  Results accumulate across frames and are averaged.
    void enableProfiling() { profilingEnabled_ = true; }

    // Stops the per-frame timestamp readback, which makes each step()
    // synchronous, right away. Compute graphs added afterwards no longer
    // record timestamps.
    void disableProfiling() { profilingEnabled_ = false; }

    bool isProfilingEnabled() const { return profilingEnabled_; }

    // Call before add() to enable VK_KHR_performance_query counter profiling.
    // nameFilter: sub-strings matched against counter name/description; empty = all counters.
    void enablePerformanceProfiling(std::vector<std::string> nameFilter = {}) {
        perfProfilingEnabled_   = true;
        perfProfilingNameFilter_ = std::move(nameFilter);
    }

    // Waits for the GPU to be idle, reads the last frame's timestamps, and
    // returns {elementName, meanTimeMs} for every element across all graphs,
    // followed by {elementName " [counterName]", value} for any perf counters.
    std::vector<std::pair<std::string, float>> getProfilingResults();

private:
    // Recreates the swapchain and rebuilds all graphs via graphBuilder_.
    // Returns false when the surface currently has zero size (minimized).
    bool rebuildForSwapChain();

    GraphBuilder graphBuilder_;

    bool profilingEnabled_            = false;
    bool perfProfilingEnabled_        = false;
    std::vector<std::string> perfProfilingNameFilter_;

    VulkanContext vulkanContext;

    std::shared_ptr<InterfaceCamera> interfaceCamera;
    std::shared_ptr<CameraUboType> cameraUBO;

    std::queue<std::unique_ptr<Event> > eventQueue;

    std::vector<std::unique_ptr<ComputeGraph>> computeGraphs;

    std::unique_ptr<Window> window_;

    std::shared_ptr<FrameOverlay> overlay_;

};

} // namespace klartraum

#endif // KLARTRAUM_CORE_HPP