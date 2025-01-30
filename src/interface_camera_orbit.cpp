#include <chrono>
#include <glm/gtc/matrix_transform.hpp>

#include "klartraum/interface_camera_orbit.hpp"
#include "klartraum/backend_vulkan.hpp"

namespace klartraum {

InterfaceCameraOrbit::InterfaceCameraOrbit(BackendVulkan *backend) : backend(backend)
{
}

void InterfaceCameraOrbit::update(Camera &camera)
{
    auto swapChainExtent = backend->getSwapChainExtent();

    static auto startTime = std::chrono::high_resolution_clock::now();
    auto currentTime = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

    auto& ubo = camera.ubo;

    ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));

    ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));

    ubo.proj = glm::perspective(glm::radians(45.0f), swapChainExtent.width / (float) swapChainExtent.height, 0.1f, 10.0f);

    // Vulkan has inverted Y coordinates compared to OpenGL
    ubo.proj[1][1] *= -1;


}

} // namespace klartraum