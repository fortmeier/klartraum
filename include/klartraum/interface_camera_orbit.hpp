#ifndef KLARTRAUM_INTERFACE_CAMERA_ORBIT_HPP
#define KLARTRAUM_INTERFACE_CAMERA_ORBIT_HPP

#include <unordered_map>

#include "klartraum/interface_camera.hpp"

namespace klartraum {

class InterfaceCameraOrbit : public InterfaceCamera {
public:
    enum class UpDirection {
        Z,
        Y
    };    

    InterfaceCameraOrbit(UpDirection up = UpDirection::Z);

    virtual void initialize(VulkanContext& vulkanContext) override;

    virtual void update(CameraMVP& mvp);
    virtual void onEvent(Event& event);

    void updatePosition(float deltaTime);


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

    void setPosition(const glm::vec3& position) {
        this->position = position;
    }

    void setNearPlane(float nearPlane) {
        this->nearPlane = nearPlane;
    }

    void setFarPlane(float farPlane) {
        this->farPlane = farPlane;
    }

    float getNearPlane() const {
        return nearPlane;
    }

    float getFarPlane() const {
        return farPlane;
    }


private:
    VulkanContext* vulkanContext;
    double azimuth = 0.0;
    double elevation = 0.0;
    double distance = 2.0;

    glm::vec3 position = glm::vec3(0.0f);

    float nearPlane = 0.1f;
    float farPlane = 1000.0f;

    bool leftButtonDown = false;

    UpDirection up = UpDirection::Z;

    std::unordered_map<EventKey::Key, bool> pressedKeys;
};

} // namespace klartraum

#endif // KLARTRAUM_INTERFACE_CAMERA_ORBIT_HPP