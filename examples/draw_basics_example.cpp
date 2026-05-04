#include <iostream>

#ifdef _WIN32
#include <windows.h>
#include <crtdbg.h>
#endif

#include "klartraum/glfw_frontend.hpp"
#include "klartraum/draw_basics.hpp"
#include "klartraum/interface_camera_orbit.hpp"

int main() {
#ifdef _WIN32
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportMode(_CRT_ERROR,  _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
    _CrtSetReportFile(_CRT_ERROR,  _CRTDBG_FILE_STDERR);
#endif
    std::cout << "DrawBasics example" << std::endl;

    klartraum::GlfwFrontend frontend;
    auto& engine = frontend.getKlartraumEngine();

    klartraum::RenderPassPtr renderpass = engine.createRenderPass();

    auto axes = std::make_shared<klartraum::DrawBasics>(klartraum::DrawBasicsType::Axes);
    renderpass->addDrawComponent(axes);

    engine.add(renderpass);

    auto cameraUBO = renderpass->getCameraUBO();
    cameraUBO->setName("CameraUBO");

    auto cameraOrbit = std::make_shared<klartraum::InterfaceCameraOrbit>(
        klartraum::InterfaceCameraOrbit::UpDirection::Y);
    cameraOrbit->setDistance(3.0);
    engine.setInterfaceCamera(cameraOrbit);
    engine.setCameraUBO(cameraUBO);

    frontend.loop();

    return 0;
}
