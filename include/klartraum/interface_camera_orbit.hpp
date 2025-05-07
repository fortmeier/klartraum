#ifndef KLARTRAUM_INTERFACE_CAMERA_ORBIT_HPP
#define KLARTRAUM_INTERFACE_CAMERA_ORBIT_HPP

#include "klartraum/interface_camera.hpp"

namespace klartraum {

class InterfaceCameraOrbit : public InterfaceCamera {
public:
    enum class UpDirection {
        Z,
        Y
    };    

    InterfaceCameraOrbit(UpDirection up = UpDirection::Z);

    virtual void initialize(VulkanKernel& vulkanKernel) override;

    virtual void update(CameraMVP& mvp);
    virtual void onEvent(Event& event);


    void setUpDirection(UpDirection up) {
        this->up = up;
    }

    void setDistance(double distance) {
        this->distance = distance;
    }

    void setAzimuth(double azimuth) {
        this->azimuth = azimuth;
    }

    void setElevation(double elevation) {
        this->elevation = elevation;
    }


private:
    VulkanKernel* vulkanKernel;
    double azimuth = 0.0;
    double elevation = 0.0;
    double distance = 2.0;

    bool leftButtonDown = false;

    UpDirection up = UpDirection::Z;
};

} // namespace klartraum

#endif // KLARTRAUM_INTERFACE_CAMERA_ORBIT_HPP