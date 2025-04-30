#ifndef KLARTRAUM_CAMERA_HPP
#define KLARTRAUM_CAMERA_HPP

#include <glm/glm.hpp>


namespace klartraum {

struct CameraMVP{
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
};



} // namespace klartraum

#endif // KLARTRAUM_CAMERA_HPP