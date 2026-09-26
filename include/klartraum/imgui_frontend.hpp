#ifndef KLARTRAUM_IMGUI_FRONTEND_HPP
#define KLARTRAUM_IMGUI_FRONTEND_HPP

#include <functional>
#include <memory>

#include "klartraum/glfw_frontend.hpp"
#include "klartraum/imgui_overlay.hpp"

namespace klartraum {

// A GlfwFrontend with a Dear ImGui user interface drawn on top of every
// frame. Set the UI with setGui(); the callback runs once per frame between
// ImGui's NewFrame and Render, so it may issue any ImGui:: calls. Mouse and
// keyboard input that ImGui uses is not forwarded to the engine (camera).
class ImGuiFrontend : public GlfwFrontend {
public:
    ImGuiFrontend();
    ~ImGuiFrontend() override;

    using GuiCallback = std::function<void()>;
    void setGui(GuiCallback gui);

    ImGuiOverlay& getOverlay() { return *overlay; }

protected:
    void beforeStep() override;
    bool wantCaptureMouse() const override;
    bool wantCaptureKeyboard() const override;

private:
    std::shared_ptr<ImGuiOverlay> overlay;
    GuiCallback gui;
};

} // namespace klartraum

#endif // KLARTRAUM_IMGUI_FRONTEND_HPP
