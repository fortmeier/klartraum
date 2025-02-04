#ifndef BACKEND_VULKAN_HPP
#define BACKEND_VULKAN_HPP

#include <vector>
#include <queue>
#include <optional>
#include <memory>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "klartraum/backend_config.hpp"
#include "klartraum/draw_component.hpp"
#include "klartraum/camera.hpp"
#include "klartraum/interface_camera.hpp"
#include "klartraum/events.hpp"

#include "klartraum/vulkan_kernel.hpp"
#include "klartraum/klartraum_core.hpp"

namespace klartraum {

class GlfwFrontend {
/**
 * @brief User facing class
 * 
 */
public:
    GlfwFrontend();
    ~GlfwFrontend();

    void initialize();

    void loop();

    void shutdown();

    void processGLFWEvents();

    KlartraumCore& getKlartraumCore();

private:
    GLFWwindow* window;
    VkSurfaceKHR surface;

    int old_mouse_x = 0;
    int old_mouse_y = 0;

    bool leftButtonDown = false;
    bool rightButtonDown = false;

    std::unique_ptr<KlartraumCore> klartraumCore;

};

} // namespace klartraum

#endif // BACKEND_VULKAN_HPP