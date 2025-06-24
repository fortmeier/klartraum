#ifndef KLARTRAUM_INTERFACE_CAMERA_HPP
#define KLARTRAUM_INTERFACE_CAMERA_HPP

#include "klartraum/camera.hpp"
#include "klartraum/events.hpp"
#include "klartraum/vulkan_context.hpp"

namespace klartraum {

class InterfaceCamera {
public:

    virtual void initialize(VulkanContext& vulkanContext) = 0;

    virtual void update(CameraMVP& mvp) = 0;
    virtual void onEvent(Event& event) {}
};

} // namespace klartraum


#endif // KLARTRAUM_INTERFACE_CAMERA_HPP