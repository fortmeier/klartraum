#ifndef KLARTRAUM_FRAME_OVERLAY_HPP
#define KLARTRAUM_FRAME_OVERLAY_HPP

#include <vulkan/vulkan.h>

namespace klartraum {

// Per-frame work that draws on top of the finished swapchain image, e.g. an
// immediate-mode GUI. Unlike the compute graphs, whose command buffers are
// recorded once and resubmitted unchanged, an overlay records its commands
// anew every frame, so its content may change freely.
//
// KlartraumEngine::step() submits the overlay after all graphs (or after the
// viewport composite) and before presenting. At that point the swapchain
// image is in PRESENT_SRC_KHR (GENERAL when headless) and must be left in the
// same layout. This requires at least one graph (or the viewport composite)
// to have rendered the image.
class FrameOverlay {
public:
    virtual ~FrameOverlay() = default;

    // Records and submits the overlay for swapchain image `imageIndex`.
    // `frameIndex` is the frame-in-flight slot, whose previous submission has
    // completed (its fence was waited on in beginRender), so per-frame
    // resources indexed by it may be reused. The submission waits on
    // `waitSemaphore`, signals `fence`, and returns a semaphore that is
    // signaled once the overlay has finished; the engine presents on it.
    virtual VkSemaphore submit(VkQueue queue, uint32_t imageIndex, uint32_t frameIndex,
                               VkSemaphore waitSemaphore, VkFence fence) = 0;

    // Called after the swapchain has been recreated (e.g. window resize);
    // everything recorded against the old swapchain images must be rebuilt.
    virtual void onSwapChainRecreated() = 0;
};

} // namespace klartraum

#endif // KLARTRAUM_FRAME_OVERLAY_HPP
