#include <stdexcept>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>


#include "klartraum/glfw_frontend.hpp"
#include "klartraum/events.hpp"

namespace klartraum {


GlfwFrontend::GlfwFrontend() : old_mouse_x(0), old_mouse_y(0)
{
    // GLFW needs to be initialized before Vulkan
    // which is done in the core afterwards
    if (!glfwInit()) {
        throw std::runtime_error("Failed to initialize GLFW");
    }

    klartraumCore = std::make_unique<KlartraumCore>();

    initialize();
}

GlfwFrontend::~GlfwFrontend()
{
    shutdown();
    glfwTerminate();
}

static double scrollYAccum = 0.0;
static double scrollXAccum = 0.0;

static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    scrollXAccum += xoffset;
    scrollYAccum += yoffset;
}


void GlfwFrontend::initialize() {
    // Initialize the glfw window

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    
    // TODO window size should be configurable somewhere else
    auto config = klartraumCore->getVulkanKernel().getConfig();
    window = glfwCreateWindow(config.WIDTH, config.HEIGHT, "Klartraum Engine", nullptr, nullptr);

    // set GLFW event callbacks
    glfwSetScrollCallback(window, scroll_callback);

    auto instance = klartraumCore->getVulkanKernel().getInstance();

    // cerate the window surface
    // (this needs to be done after the Vulkan instance is created)
    if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
        throw std::runtime_error("failed to create window surface!");
    }

    klartraumCore->getVulkanKernel().initialize(surface);
}


void GlfwFrontend::loop() {

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        processGLFWEvents();

        klartraumCore->step();
    }

}

void GlfwFrontend::shutdown() {
    auto& instance = klartraumCore->getVulkanKernel().getInstance();

    klartraumCore->getVulkanKernel().shutdown();

    vkDestroySurfaceKHR(instance, surface, nullptr);

    glfwDestroyWindow(window);
    glfwTerminate();
}


void GlfwFrontend::processGLFWEvents() {
    auto& eventQueue = klartraumCore->getEventQueue();

    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);
    int new_mouse_x = (int)xpos;
    int new_mouse_y = (int)ypos;
    if(new_mouse_x != old_mouse_x || new_mouse_y != old_mouse_y) {
        int dx = new_mouse_x - old_mouse_x;
        int dy = new_mouse_y - old_mouse_y;
        auto event = std::make_unique<EventMouseMove>(new_mouse_x, new_mouse_y, dx, dy);
        eventQueue.push(std::move(event));
        old_mouse_x = new_mouse_x;
        old_mouse_y = new_mouse_y;
    }

    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS && !leftButtonDown) {
        auto event = std::make_unique<EventMouseButton>(EventMouseButton::Button::Left, EventMouseButton::Action::Press);
        eventQueue.push(std::move(event));
        leftButtonDown = true;
    }
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_RELEASE && leftButtonDown) {
        auto event = std::make_unique<EventMouseButton>(EventMouseButton::Button::Left, EventMouseButton::Action::Release);
        eventQueue.push(std::move(event));
        leftButtonDown = false;
    }

    if (scrollXAccum != 0.0 || scrollYAccum != 0.0) {
        auto event = std::make_unique<EventMouseScroll>(scrollXAccum, scrollYAccum);
        eventQueue.push(std::move(event));
        scrollXAccum = 0.0;
        scrollYAccum = 0.0;
    }

}

KlartraumCore& GlfwFrontend::getKlartraumCore()
{
    return *klartraumCore;
}

} // namespace klartraum