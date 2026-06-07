#ifndef KLARTRAUM_CAMERA_HPP
#define KLARTRAUM_CAMERA_HPP

#include <glm/glm.hpp>


namespace klartraum {

struct CameraMVP{
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
    // World-space camera position (xyz); w unused. Stored as vec4 so its
    // std140 layout matches glm::vec4 exactly on both the C++ and GLSL sides.
    glm::vec4 cameraWorldPos;
};



} // namespace klartraum

#endif // KLARTRAUM_CAMERA_HPP