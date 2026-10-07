// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_CAMERA_HPP
#define KLARTRAUM_CAMERA_HPP

#include <glm/glm.hpp>

namespace klartraum {

/**
 * @brief Contents of the camera uniform buffer (CameraUboType): model, view and projection matrices and the camera
 * position, in std140 layout.
 */
struct CameraMVP {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
    // World-space camera position (xyz); w unused. Stored as vec4 so its
    // std140 layout matches glm::vec4 exactly on both the C++ and GLSL sides.
    glm::vec4 cameraWorldPos;
};

} // namespace klartraum

#endif // KLARTRAUM_CAMERA_HPP
