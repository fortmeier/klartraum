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

/**
 * @brief A projection that shows only a window of `proj`'s view, enlarged to the whole view.
 *
 * The window is the square of half extent `halfExtent` around `centre`, both in
 * the NDC of `proj`; NDC depths stay the same. Rendering with it at a
 * resolution of the window's size in pixels gives the window's pixels of the
 * full view; at a higher resolution, a sharper view of the window.
 */
inline glm::mat4 cropProjection(const glm::mat4& proj, glm::vec2 centre, float halfExtent) {
    // x' = (x - centre.x * w) / halfExtent, likewise y; z and w stay.
    glm::mat4 crop(1.0f);
    crop[0][0] = 1.0f / halfExtent;
    crop[1][1] = 1.0f / halfExtent;
    crop[3][0] = -centre.x / halfExtent;
    crop[3][1] = -centre.y / halfExtent;
    return crop * proj;
}

} // namespace klartraum

#endif // KLARTRAUM_CAMERA_HPP
