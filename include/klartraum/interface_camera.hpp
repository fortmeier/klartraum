#ifndef KLARTRAUM_INTERFACE_CAMERA_HPP
#define KLARTRAUM_INTERFACE_CAMERA_HPP

#include "klartraum/camera.hpp"
#include "klartraum/events.hpp"

namespace klartraum {

class InterfaceCamera {
public:
    virtual void update(Camera& camera) = 0;
    virtual void onEvent(Event& event) {}
};

} // namespace klartraum


#endif // KLARTRAUM_INTERFACE_CAMERA_HPP