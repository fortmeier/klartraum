#include "klartraum/imgui_frontend.hpp"

#include <stdexcept>

#include <imgui.h>
#include <imgui_impl_glfw.h>

namespace klartraum {

ImGuiFrontend::ImGuiFrontend() {
    auto& engine = getKlartraumEngine();

    // Creates the ImGui context and makes it current.
    overlay = std::make_shared<ImGuiOverlay>(engine.getVulkanContext());

    // Installs ImGui's GLFW callbacks, chained to the ones GlfwFrontend set,
    // so the frontend still receives scroll and key events.
    if (!ImGui_ImplGlfw_InitForVulkan(getGlfwWindow(), true)) {
        throw std::runtime_error("failed to initialize the ImGui GLFW backend!");
    }

    engine.setOverlay(overlay);
}

ImGuiFrontend::~ImGuiFrontend() {
    auto& engine = getKlartraumEngine();
    vkDeviceWaitIdle(engine.getVulkanContext().getDevice());
    engine.setOverlay(nullptr);

    // The GLFW backend lives in the overlay's ImGui context, so it goes first.
    ImGui_ImplGlfw_Shutdown();
    overlay.reset();
}

void ImGuiFrontend::setGui(GuiCallback gui) {
    this->gui = std::move(gui);
}

void ImGuiFrontend::beforeStep() {
    ImGui_ImplGlfw_NewFrame();
    overlay->newFrame();
    if (gui) {
        gui();
    }
    overlay->render();
}

bool ImGuiFrontend::wantCaptureMouse() const {
    return ImGui::GetIO().WantCaptureMouse;
}

bool ImGuiFrontend::wantCaptureKeyboard() const {
    return ImGui::GetIO().WantCaptureKeyboard;
}

} // namespace klartraum
