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

    virtual void update(Camera& camera);
    virtual void onEvent(Event& event);


    void setUpDirection(UpDirection up) {
        this->up = up;
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