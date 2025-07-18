#include <iostream>
#include <chrono>
#include <glm/gtc/matrix_transform.hpp>

#include "klartraum/interface_camera_orbit.hpp"
#include "klartraum/glfw_frontend.hpp"

namespace klartraum {

InterfaceCameraOrbit::InterfaceCameraOrbit(UpDirection up)
{
    setUpDirection(up);
}

void InterfaceCameraOrbit::initialize(VulkanContext& vulkanContext) {
    this->vulkanContext = &vulkanContext;
}

void InterfaceCameraOrbit::update(CameraMVP &mvp)
{
    auto swapChainExtent = vulkanContext->getSwapChainExtent();

    static auto startTime = std::chrono::high_resolution_clock::now();
    auto currentTime = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

    updatePosition(time);

    mvp.model = glm::mat4(1.0f); //glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));

    switch (up)
    {
    case UpDirection::Y:
        mvp.view = 
                glm::lookAt(glm::vec3(distance, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f)) *
                glm::rotate(glm::mat4(1.0f), (float)elevation, glm::vec3(0.0f, 0.0f, 1.0f)) *    
                glm::rotate(glm::mat4(1.0f), (float)azimuth, glm::vec3(0.0f, 1.0f, 0.0f))
                * glm::translate(glm::mat4(1.0f), position)
                ;
        break;

    case UpDirection::Z:
        mvp.view = glm::lookAt(glm::vec3(distance, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f)) *
                glm::rotate(glm::mat4(1.0f), (float)elevation, glm::vec3(0.0f, 1.0f, 0.0f)) *    
                glm::rotate(glm::mat4(1.0f), (float)azimuth, glm::vec3(0.0f, 0.0f, 1.0f));
        break;
    default:
        throw std::runtime_error("Unknown up direction");
        break;
    }
    
    mvp.proj = glm::perspective(glm::radians(45.0f), vulkanContext->getSwapChainExtent().width / (float) swapChainExtent.height, 0.1f, 10.0f);

    // Vulkan has inverted Y coordinates compared to OpenGL
    mvp.proj[1][1] *= -1;
}

void InterfaceCameraOrbit::updatePosition(float deltaTime)
{
    float moveSpeed = 0.01f * distance;
    glm::vec3 forward, right, upVec;

    switch (up) {
    case UpDirection::Y:
        forward = glm::vec3(
            -cos(azimuth) * cos(elevation),
            -sin(azimuth) * cos(elevation),
            sin(elevation)
        );
        upVec = glm::vec3(0.0f, 1.0f, 0.0f);
        right = glm::normalize(glm::cross(forward, upVec));
        break;
    case UpDirection::Z:
        forward = glm::vec3(
            -cos(azimuth) * cos(elevation),
            sin(elevation),
            -sin(azimuth) * cos(elevation)
        );
        upVec = glm::vec3(0.0f, 0.0f, 1.0f);
        right = glm::normalize(glm::cross(forward, upVec));
        break;
    default:
        forward = glm::vec3(0.0f, 0.0f, 0.0f);
        right = glm::vec3(0.0f, 0.0f, 0.0f);
        upVec = glm::vec3(0.0f, 1.0f, 0.0f);
        break;
    }

    if (pressedKeys[EventKey::Key::W]) {
        position += forward * moveSpeed;
    }
    if (pressedKeys[EventKey::Key::S]) {
        position -= forward * moveSpeed;
    }
    if (pressedKeys[EventKey::Key::A]) {
        position += right * moveSpeed;
    }
    if (pressedKeys[EventKey::Key::D]) {
        position -= right * moveSpeed;
    }

}

void InterfaceCameraOrbit::onEvent(Event &event)
{
    static auto startTime = std::chrono::high_resolution_clock::now();
    auto currentTime = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

    if (EventMouseMove *e = dynamic_cast<EventMouseMove *>(&event))
    {
        if(leftButtonDown)
        {
            azimuth += e->dx / 1000.0f * time;
            elevation += e->dy / 1000.0f * time;

            if (elevation > glm::radians(89.0f))
                elevation = glm::radians(89.0f);
            if (elevation < glm::radians(-89.0f))
                elevation = glm::radians(-89.0f);

            if (azimuth > glm::radians(360.0f))
                azimuth -= glm::radians(360.0f);
            if (azimuth < glm::radians(0.0f))
                azimuth += glm::radians(360.0f);
        }
    }
    else if(EventMouseButton *e = dynamic_cast<EventMouseButton *>(&event))
    {
        if(e->button == EventMouseButton::Button::Left )
        {
            leftButtonDown = e->action == EventMouseButton::Action::Press;
        }

    }
    else if(EventMouseScroll *e = dynamic_cast<EventMouseScroll *>(&event))
    {
        distance += e->y / 10.0;
        distance = glm::clamp(distance, 0.1, 10000.0);
    }
    else if(EventKey *e = dynamic_cast<EventKey *>(&event))
    {
        if(e->key == EventKey::Key::Escape)
        {
            // Handle escape key press
            exit(0);
        }
        else if (e->key == EventKey::Key::Space)
        {
            // Reset camera position and orientation
            azimuth = 0.0;
            elevation = 0.0;
            distance = 2.0;
            position = glm::vec3(0.0f);
        }
        else
        {
            pressedKeys[e->key] = (e->action == EventKey::Action::Press);
        }
    }

}

} // namespace klartraum