#ifndef KLARTRAUM_INTERFACE_CAMERA_ORBIT_HPP
#define KLARTRAUM_INTERFACE_CAMERA_ORBIT_HPP

#include "klartraum/interface_camera.hpp"

namespace klartraum {

class InterfaceCameraOrbit : public InterfaceCamera {
public:
    InterfaceCameraOrbit(BackendVulkan* backend);

    virtual void update(Camera& camera);
    virtual void onEvent(Event& event);

    enum class UpDirection {
        Z
    };    

    void setUpDirection(UpDirection up) {
        this->up = up;
    }


private:
    BackendVulkan* backend;
    double azimuth = 0.0;
    double elevation = 0.0;

    bool leftButtonDown = false;

    UpDirection up = UpDirection::Z;
};

} // namespace klartraum

#endif // KLARTRAUM_INTERFACE_CAMERA_ORBIT_HPP