#ifndef BACKEND_VULKAN_HPP
#define BACKEND_VULKAN_HPP

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

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


    void loop();



    KlartraumCore& getKlartraumCore();

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

    std::unique_ptr<KlartraumCore> klartraumCore;

};

} // namespace klartraum

#endif // BACKEND_VULKAN_HPP