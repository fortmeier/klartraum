#ifndef KLARTRAUM_INTERFACE_CAMERA_HPP
#define KLARTRAUM_INTERFACE_CAMERA_HPP

#include "klartraum/camera.hpp"

namespace klartraum {

class InterfaceCamera {
public:
    virtual void update(Camera& camera) = 0;
};

} // namespace klartraum


#endif // KLARTRAUM_INTERFACE_CAMERA_HPP