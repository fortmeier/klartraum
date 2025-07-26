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

    klartraumEngine = std::make_unique<KlartraumEngine>();

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

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    auto frontend = static_cast<GlfwFrontend*>(glfwGetWindowUserPointer(window));
    frontend->keyCallback(window, key, scancode, action, mods);
}

void GlfwFrontend::initialize() {
    // Initialize the glfw window

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    
    // TODO window size should be configurable somewhere else
    auto config = klartraumEngine->getVulkanContext().getConfig();
    window = glfwCreateWindow(config.WIDTH, config.HEIGHT, config.ENGINE_VERSION, nullptr, nullptr);
    
    float xscale, yscale;
    glfwGetWindowContentScale(window, &xscale, &yscale);

    // If the content scale is not 1.0, we need to adjust the window size
    if(xscale != 1.0f || yscale != 1.0f) {
        glfwDestroyWindow(window);
        window = glfwCreateWindow(config.WIDTH / xscale, config.HEIGHT / yscale, config.ENGINE_VERSION, nullptr, nullptr);
    }

    glfwSetWindowUserPointer(window, this);

    // set GLFW event callbacks
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetKeyCallback(window, key_callback);

    auto instance = klartraumEngine->getVulkanContext().getInstance();

    // cerate the window surface
    // (this needs to be done after the Vulkan instance is created)
    if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
        throw std::runtime_error("failed to create window surface!");
    }

    klartraumEngine->getVulkanContext().initialize(surface);
}


void GlfwFrontend::loop() {

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        processGLFWEvents();

        klartraumEngine->step();
    }

}

void GlfwFrontend::shutdown() {
    auto& instance = klartraumEngine->getVulkanContext().getInstance();
    auto& vulkanContext = klartraumEngine->getVulkanContext();
    
    vulkanContext.stopRender();
    klartraumEngine->clearComputeGraphs();
    vulkanContext.shutdown();

    vkDestroySurfaceKHR(instance, surface, nullptr);

    glfwDestroyWindow(window);
    glfwTerminate();
}


void GlfwFrontend::processGLFWEvents() {
    auto& eventQueue = klartraumEngine->getEventQueue();

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

constexpr EventKey::Key translateKey(int glfwKey) {
    switch (glfwKey) {
        case GLFW_KEY_SPACE: return EventKey::Key::Space;
        case GLFW_KEY_APOSTROPHE: return EventKey::Key::Apostrophe;
        case GLFW_KEY_COMMA: return EventKey::Key::Comma;
        case GLFW_KEY_MINUS: return EventKey::Key::Minus;
        case GLFW_KEY_PERIOD: return EventKey::Key::Period;
        case GLFW_KEY_SLASH: return EventKey::Key::Slash;
        case GLFW_KEY_0: return EventKey::Key::Num0;
        case GLFW_KEY_1: return EventKey::Key::Num1;
        case GLFW_KEY_2: return EventKey::Key::Num2;
        case GLFW_KEY_3: return EventKey::Key::Num3;
        case GLFW_KEY_4: return EventKey::Key::Num4;
        case GLFW_KEY_5: return EventKey::Key::Num5;
        case GLFW_KEY_6: return EventKey::Key::Num6;
        case GLFW_KEY_7: return EventKey::Key::Num7;
        case GLFW_KEY_8: return EventKey::Key::Num8;
        case GLFW_KEY_9: return EventKey::Key::Num9;
        case GLFW_KEY_SEMICOLON: return EventKey::Key::Semicolon;
        case GLFW_KEY_EQUAL: return EventKey::Key::Equal;
        case GLFW_KEY_A: return EventKey::Key::A;
        case GLFW_KEY_B: return EventKey::Key::B;
        case GLFW_KEY_C: return EventKey::Key::C;
        case GLFW_KEY_D: return EventKey::Key::D;
        case GLFW_KEY_E: return EventKey::Key::E;
        case GLFW_KEY_F: return EventKey::Key::F;
        case GLFW_KEY_G: return EventKey::Key::G;
        case GLFW_KEY_H: return EventKey::Key::H;
        case GLFW_KEY_I: return EventKey::Key::I;
        case GLFW_KEY_J: return EventKey::Key::J;
        case GLFW_KEY_K: return EventKey::Key::K;
        case GLFW_KEY_L: return EventKey::Key::L;
        case GLFW_KEY_M: return EventKey::Key::M;
        case GLFW_KEY_N: return EventKey::Key::N;
        case GLFW_KEY_O: return EventKey::Key::O;
        case GLFW_KEY_P: return EventKey::Key::P;
        case GLFW_KEY_Q: return EventKey::Key::Q;
        case GLFW_KEY_R: return EventKey::Key::R;
        case GLFW_KEY_S: return EventKey::Key::S;
        case GLFW_KEY_T: return EventKey::Key::T;
        case GLFW_KEY_U: return EventKey::Key::U;
        case GLFW_KEY_V: return EventKey::Key::V;
        case GLFW_KEY_W: return EventKey::Key::W;
        case GLFW_KEY_X: return EventKey::Key::X;
        case GLFW_KEY_Y: return EventKey::Key::Y;
        case GLFW_KEY_Z: return EventKey::Key::Z;
        case GLFW_KEY_LEFT_BRACKET: return EventKey::Key::LeftBracket;
        case GLFW_KEY_BACKSLASH: return EventKey::Key::Backslash;
        case GLFW_KEY_RIGHT_BRACKET: return EventKey::Key::RightBracket;
        case GLFW_KEY_GRAVE_ACCENT: return EventKey::Key::GraveAccent;
        case GLFW_KEY_WORLD_1: return EventKey::Key::World1;
        case GLFW_KEY_WORLD_2: return EventKey::Key::World2;
        case GLFW_KEY_ESCAPE: return EventKey::Key::Escape;
        case GLFW_KEY_ENTER: return EventKey::Key::Enter;
        case GLFW_KEY_TAB: return EventKey::Key::Tab;
        case GLFW_KEY_BACKSPACE: return EventKey::Key::Backspace;
        case GLFW_KEY_INSERT: return EventKey::Key::Insert;
        case GLFW_KEY_DELETE: return EventKey::Key::Delete;
        case GLFW_KEY_RIGHT: return EventKey::Key::Right;
        case GLFW_KEY_LEFT: return EventKey::Key::Left;
        case GLFW_KEY_DOWN: return EventKey::Key::Down;
        case GLFW_KEY_UP: return EventKey::Key::Up;
        case GLFW_KEY_PAGE_UP: return EventKey::Key::PageUp;
        case GLFW_KEY_PAGE_DOWN: return EventKey::Key::PageDown;
        case GLFW_KEY_HOME: return EventKey::Key::Home;
        case GLFW_KEY_END: return EventKey::Key::End;
        case GLFW_KEY_CAPS_LOCK: return EventKey::Key::CapsLock;
        case GLFW_KEY_SCROLL_LOCK: return EventKey::Key::ScrollLock;
        case GLFW_KEY_NUM_LOCK: return EventKey::Key::NumLock;
        case GLFW_KEY_PRINT_SCREEN: return EventKey::Key::PrintScreen;
        case GLFW_KEY_PAUSE: return EventKey::Key::Pause;
        case GLFW_KEY_F1: return EventKey::Key::F1;
        case GLFW_KEY_F2: return EventKey::Key::F2;
        case GLFW_KEY_F3: return EventKey::Key::F3;
        case GLFW_KEY_F4: return EventKey::Key::F4;
        case GLFW_KEY_F5: return EventKey::Key::F5;
        case GLFW_KEY_F6: return EventKey::Key::F6;
        case GLFW_KEY_F7: return EventKey::Key::F7;
        case GLFW_KEY_F8: return EventKey::Key::F8;
        case GLFW_KEY_F9: return EventKey::Key::F9;
        case GLFW_KEY_F10: return EventKey::Key::F10;
        case GLFW_KEY_F11: return EventKey::Key::F11;
        case GLFW_KEY_F12: return EventKey::Key::F12;
        case GLFW_KEY_F13: return EventKey::Key::F13;
        case GLFW_KEY_F14: return EventKey::Key::F14;
        case GLFW_KEY_F15: return EventKey::Key::F15;
        case GLFW_KEY_F16: return EventKey::Key::F16;
        case GLFW_KEY_F17: return EventKey::Key::F17;
        case GLFW_KEY_F18: return EventKey::Key::F18;
        case GLFW_KEY_F19: return EventKey::Key::F19;
        case GLFW_KEY_F20: return EventKey::Key::F20;
        case GLFW_KEY_F21: return EventKey::Key::F21;
        case GLFW_KEY_F22: return EventKey::Key::F22;
        case GLFW_KEY_F23: return EventKey::Key::F23;
        case GLFW_KEY_F24: return EventKey::Key::F24;
        case GLFW_KEY_F25: return EventKey::Key::F25;
        case GLFW_KEY_KP_0: return EventKey::Key::KP_0;
        case GLFW_KEY_KP_1: return EventKey::Key::KP_1;
        case GLFW_KEY_KP_2: return EventKey::Key::KP_2;
        case GLFW_KEY_KP_3: return EventKey::Key::KP_3;
        case GLFW_KEY_KP_4: return EventKey::Key::KP_4;
        case GLFW_KEY_KP_5: return EventKey::Key::KP_5;
        case GLFW_KEY_KP_6: return EventKey::Key::KP_6;
        case GLFW_KEY_KP_7: return EventKey::Key::KP_7;
        case GLFW_KEY_KP_8: return EventKey::Key::KP_8;
        case GLFW_KEY_KP_9: return EventKey::Key::KP_9;
        case GLFW_KEY_KP_DECIMAL: return EventKey::Key::KP_Decimal;
        case GLFW_KEY_KP_DIVIDE: return EventKey::Key::KP_Divide;
        case GLFW_KEY_KP_MULTIPLY: return EventKey::Key::KP_Multiply;
        case GLFW_KEY_KP_SUBTRACT: return EventKey::Key::KP_Subtract;
        case GLFW_KEY_KP_ADD: return EventKey::Key::KP_Add;
        case GLFW_KEY_KP_ENTER: return EventKey::Key::KP_Enter;
        case GLFW_KEY_KP_EQUAL: return EventKey::Key::KP_Equal;
        case GLFW_KEY_LEFT_SHIFT: return EventKey::Key::LeftShift;
        case GLFW_KEY_LEFT_CONTROL: return EventKey::Key::LeftControl;
        case GLFW_KEY_LEFT_ALT: return EventKey::Key::LeftAlt;
        case GLFW_KEY_LEFT_SUPER: return EventKey::Key::LeftSuper;
        case GLFW_KEY_RIGHT_SHIFT: return EventKey::Key::RightShift;
        case GLFW_KEY_RIGHT_CONTROL: return EventKey::Key::RightControl;
        case GLFW_KEY_RIGHT_ALT: return EventKey::Key::RightAlt;
        case GLFW_KEY_RIGHT_SUPER: return EventKey::Key::RightSuper;
        case GLFW_KEY_MENU: return EventKey::Key::Menu;
        default: return EventKey::Key::Unknown;
    }
}

void GlfwFrontend::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    auto& eventQueue = klartraumEngine->getEventQueue();
    EventKey::Key klartraumKey = translateKey(key);

    auto event = std::make_unique<EventKey>(klartraumKey, action == GLFW_PRESS ? EventKey::Action::Press : EventKey::Action::Release);
    eventQueue.push(std::move(event));
}

KlartraumEngine& GlfwFrontend::getKlartraumEngine()
{
    return *klartraumEngine;
}

} // namespace klartraum