#ifndef BACKEND_VULKAN_HPP
#define BACKEND_VULKAN_HPP

#include <exception>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "klartraum/klartraum_core.hpp"
#include "klartraum/text_draw_component.hpp"

namespace klartraum {

/**
 * @brief Windowed frontend that owns GLFW event processing and frame submission.
 *
 * Most frames are rendered by loop() after pending input events have been
 * processed. A resize or refresh callback may also render synchronously from
 * inside glfwPollEvents(), because some window systems keep control until an
 * interactive resize finishes. Both paths go through the same frame routine
 * and count toward the same finite loop budget.
 *
 * @see loop()
 * @see renderFromEventCallback()
 */
class GlfwFrontend {
public:
    GlfwFrontend();
    virtual ~GlfwFrontend();

    /**
     * @brief Processes window events and renders frames until the window closes.
     * @param maxFrames Maximum number of frames to render, or a non-positive
     *        value to continue until the user closes the window.
     *
     * A frame rendered synchronously by a resize or refresh callback consumes
     * the same budget as a normal loop frame. If event processing already
     * rendered a frame, that loop iteration does not render a duplicate frame.
     */
    void loop(int maxFrames = -1);



    KlartraumEngine& getKlartraumEngine();

    GLFWwindow* getGlfwWindow() { return window; }

    void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);

    /**
     * @brief Dispatches pending GLFW events and rethrows callback failures.
     *
     * While a window is being resized interactively, the operating system may
     * not return from event processing until the drag ends. Resize and refresh
     * callbacks therefore may render frames before this method returns.
     * Exceptions cannot unwind through GLFW's C callbacks, so callback failures
     * are captured and rethrown here on the C++ stack.
     */
    void pollEvents();

    /**
     * @brief Renders one frame synchronously from a GLFW window callback.
     *
     * This keeps a resizable window responsive while the native event loop owns
     * control. It is called by the installed framebuffer-size and refresh
     * callbacks; applications normally do not call it directly.
     */
    void renderFromEventCallback();

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

protected:
    /**
     * @brief Hook invoked immediately before every normal or event-driven frame.
     *
     * ImGuiFrontend uses this hook to begin, build, and finalize its UI frame.
     */
    virtual void beforeStep() {}

    // While these return true, mouse (buttons, scroll) or keyboard input is
    // not forwarded to the engine's event queue, e.g. because a GUI uses it.
    virtual bool wantCaptureMouse() const { return false; }
    virtual bool wantCaptureKeyboard() const { return false; }

private:
    void initialize();
    void shutdown();
    void processGLFWEvents();
    bool renderFrame();

    GLFWwindow* window;
    VkSurfaceKHR surface;

    int old_mouse_x = 0;
    int old_mouse_y = 0;

    bool leftButtonDown = false;
    bool rightButtonDown = false;

    std::unique_ptr<KlartraumEngine> klartraumEngine;

    std::shared_ptr<TextDrawComponent> textOverlay;

    // Exceptions must not unwind through GLFW's C (and Objective-C) frames.
    std::exception_ptr callbackError;

    // Event callbacks can render while glfwPollEvents() is active. Keep those
    // frames in the loop(maxFrames) budget and avoid rendering the same loop
    // iteration a second time after event processing returns.
    int loopFramesRemaining = -1;
    bool renderedFromEventCallback = false;

};

} // namespace klartraum

#endif // BACKEND_VULKAN_HPP
