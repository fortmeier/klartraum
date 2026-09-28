#ifndef KLARTRAUM_WINDOW_HPP
#define KLARTRAUM_WINDOW_HPP

#include <memory>
#include <vector>

#include <vulkan/vulkan.h>

#include "klartraum/computegraph/imageviewsrc.hpp"
#include "klartraum/offscreen_target.hpp"

namespace klartraum {

class VulkanContext;

// Owns the swapchain's viewport composition. makeViewport() hands out offscreen
// render targets (sub-regions of the window); a single pre-recorded composite
// per swapchain image blits every viewport into its destination rectangle and
// transitions the swapchain image for presentation. The engine submits the
// composite once per frame after the viewport scenes have rendered.
class Window {
public:
    explicit Window(VulkanContext& vulkanContext);
    ~Window();

    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    // Carve a w*h viewport at (x, y) out of the window. Returns a render target
    // usable as the imageViewSrc argument of the scene factories. Must be called
    // before the first composite submission.
    std::shared_ptr<ImageViewSrc> makeViewport(int x, int y, uint32_t width, uint32_t height);

    // Create a viewport whose offscreen render resolution differs from its
    // displayed size. The composite scales the source image to the destination.
    std::shared_ptr<ImageViewSrc> makeViewport(int x, int y,
                                               uint32_t displayWidth, uint32_t displayHeight,
                                               uint32_t renderWidth, uint32_t renderHeight);

    bool hasViewports() const { return !viewports_.empty(); }

    // Submit the composite for swapchain image `imageIndex`, waiting on every
    // semaphore in `waitSemaphores` (the engine passes the viewport scenes'
    // finished semaphores plus the image-available semaphore). Signals — and
    // returns — the composite-finished semaphore for that image, and signals
    // `fence` (may be VK_NULL_HANDLE) when the composite completes.
    VkSemaphore submitComposite(VkQueue queue, uint32_t imageIndex,
                                const std::vector<VkSemaphore>& waitSemaphores,
                                VkFence fence);

private:
    void finalize();   // allocate + record composite command buffers (idempotent)

    struct Viewport {
        std::shared_ptr<OffscreenTarget> target;
        VkRect2D rect;
    };

    VulkanContext& vulkanContext_;
    std::vector<Viewport> viewports_;

    bool finalized_ = false;
    std::vector<VkCommandBuffer> composite_;          // one per swapchain image
    std::vector<VkSemaphore>     compositeFinished_;  // one per swapchain image
};

} // namespace klartraum

#endif // KLARTRAUM_WINDOW_HPP
