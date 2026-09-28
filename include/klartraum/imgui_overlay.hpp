#ifndef KLARTRAUM_IMGUI_OVERLAY_HPP
#define KLARTRAUM_IMGUI_OVERLAY_HPP

#include <vector>

#include <vulkan/vulkan.h>

#include "klartraum/frame_overlay.hpp"

struct ImGuiContext;

namespace klartraum {

class VulkanContext;

// Draws a Dear ImGui user interface on top of every frame (see FrameOverlay).
//
// Owns the ImGui context and the ImGui Vulkan renderer backend, but no
// platform (input/window) backend: a windowed frontend adds one (see
// ImGuiFrontend), while headless use sets ImGui::GetIO().DisplaySize itself.
// Only one ImGuiOverlay may exist at a time, as the ImGui Vulkan backend keeps
// global state.
//
// Per frame, call newFrame(), build the UI with ImGui:: calls, then render(),
// and finally KlartraumEngine::step(), which submits the recorded draw data.
class ImGuiOverlay : public FrameOverlay {
public:
    explicit ImGuiOverlay(VulkanContext& vulkanContext);
    ~ImGuiOverlay() override;

    ImGuiOverlay(const ImGuiOverlay&) = delete;
    ImGuiOverlay& operator=(const ImGuiOverlay&) = delete;

    // Starts a new ImGui frame. A platform backend's NewFrame, if any, must
    // be called before this.
    void newFrame();

    // Finalizes the ImGui frame; its draw data is drawn by the next submit().
    void render();

    VkSemaphore submit(VkQueue queue, uint32_t imageIndex, uint32_t frameIndex,
                       VkSemaphore waitSemaphore, VkFence fence) override;

    void onSwapChainRecreated() override;

private:
    void createRenderPass();
    void createSwapChainResources();
    void destroySwapChainResources();

    VulkanContext& vulkanContext_;
    ImGuiContext* imguiContext_ = nullptr;

    VkFormat format_ = VK_FORMAT_UNDEFINED;
    VkRenderPass renderPass_ = VK_NULL_HANDLE;
    VkCommandPool commandPool_ = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> commandBuffers_;   // one per frame in flight

    // Recreated with the swapchain.
    std::vector<VkFramebuffer> framebuffers_;       // one per swapchain image
    std::vector<VkSemaphore> finishedSemaphores_;   // one per swapchain image
};

} // namespace klartraum

#endif // KLARTRAUM_IMGUI_OVERLAY_HPP
