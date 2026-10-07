// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_INTERFACE_CAMERA_HPP
#define KLARTRAUM_INTERFACE_CAMERA_HPP

#include "klartraum/camera.hpp"
#include "klartraum/events.hpp"
#include "klartraum/vulkan_context.hpp"

namespace klartraum {

/**
 * @brief Interface for cameras that the engine updates every frame.
 *
 * KlartraumEngine::setInterfaceCamera() calls initialize() once. Every frame,
 * KlartraumEngine::step() passes the queued input events to onEvent() and then
 * calls update(), which writes the camera matrices into the data of the camera
 * uniform buffer.
 */
class InterfaceCamera {
public:
    virtual void initialize(VulkanContext& vulkanContext) = 0;

    virtual void update(CameraMVP& mvp) = 0;
    virtual void onEvent(Event& event) {}
};

} // namespace klartraum

#endif // KLARTRAUM_INTERFACE_CAMERA_HPP
