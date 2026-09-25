#ifndef BACKEND_VULKAN_HPP
#define BACKEND_VULKAN_HPP

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "klartraum/klartraum_core.hpp"
#include "klartraum/text_draw_component.hpp"

namespace klartraum {

class GlfwFrontend {
/**
 * @brief User facing class
 * 
 */
public:
    GlfwFrontend();
    ~GlfwFrontend();


    // Run the render loop.  If maxFrames > 0 the window closes automatically
    // after that many frames; pass -1 (default) to run until user closes it.
    void loop(int maxFrames = -1);



    KlartraumEngine& getKlartraumEngine();

    GLFWwindow* getGlfwWindow() { return window; }

    void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);

    // Registers a debug-text overlay on `renderPass`, added after any draw
    // components already on it so the text is rendered last (on top of the
    // main content). Call once, before passing `renderPass` to engine.add() —
    // draw components are initialized at graph-compile time. Afterwards, use
    // renderText() to set what is displayed.
    void attachTextOverlay(RenderPassPtr renderPass);

    // Updates the debug-text overlay registered via attachTextOverlay().
    // (x, y) is the top-left corner in screen pixels; `scale` multiplies the
    // glyphs' native pixel size (5x7); (r, g, b, a) tints the text.
    void renderText(const std::string& text, float x, float y, float scale = 1.0f, float r = 1.0f, float g = 1.0f,
                    float b = 1.0f, float a = 1.0f);

private:
    void initialize();
    void shutdown();
    void processGLFWEvents();

    GLFWwindow* window;
    VkSurfaceKHR surface;

    int old_mouse_x = 0;
    int old_mouse_y = 0;

    bool leftButtonDown = false;
    bool rightButtonDown = false;

    std::unique_ptr<KlartraumEngine> klartraumEngine;

    std::shared_ptr<TextDrawComponent> textOverlay;

};

} // namespace klartraum

#endif // BACKEND_VULKAN_HPP