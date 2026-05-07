#include <iostream>
#include <filesystem>
#include <string>

#ifdef _WIN32
#include <windows.h>
#include <crtdbg.h>
#endif

#include "klartraum/glfw_frontend.hpp"
#include "klartraum/vulkan_gaussian_splatting.hpp"
#include "klartraum/interface_camera_orbit.hpp"
#include "klartraum/computegraph/imageviewsrc.hpp"

int main(int argc, char** argv) {
#ifdef _WIN32
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportMode(_CRT_ERROR,  _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
    _CrtSetReportFile(_CRT_ERROR,  _CRTDBG_FILE_STDERR);
#endif

    // Parse args:  --frames N   (close after N frames)
    int maxFrames = -1;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--frames" && i + 1 < argc) {
            try { maxFrames = std::stoi(argv[i + 1]); } catch (...) {}
        }
    }
    std::cout << "Gaussian Splatting example";
    if (maxFrames > 0) std::cout << " (closing after " << maxFrames << " frames)";
    std::cout << std::endl;

    klartraum::GlfwFrontend frontend;
    auto& engine = frontend.getKlartraumEngine();
    auto& vulkanContext = engine.getVulkanContext();

    if (maxFrames > 0) engine.enableProfiling();

    uint32_t numImages = vulkanContext.getNumberOfSwapChainImages();
    auto extent = vulkanContext.getSwapChainExtent();
    std::vector<VkImageView> imageViews(numImages);
    std::vector<VkImage>     images(numImages);
    std::vector<VkExtent2D>  extents(numImages, extent);
    for (uint32_t i = 0; i < numImages; ++i) {
        imageViews[i] = vulkanContext.getImageView(i);
        images[i]     = vulkanContext.getSwapChainImage(i);
    }
    auto imageViewSrc = std::make_shared<klartraum::ImageViewSrc>(imageViews, images, extents);
    for (uint32_t i = 0; i < numImages; ++i) {
        imageViewSrc->setWaitFor(i, vulkanContext.imageAvailableSemaphoresPerImage[i]);
    }

    auto cameraUBO = std::make_shared<klartraum::CameraUboType>();
    cameraUBO->setName("CameraUBO");

    std::string spzFile = "./3rdparty/spz/samples/racoonfamily.spz";

    auto splatting = vulkanContext.create<klartraum::VulkanGaussianSplatting>(
        imageViewSrc, cameraUBO, spzFile);
    engine.add(splatting);

    auto cameraOrbit = std::make_shared<klartraum::InterfaceCameraOrbit>(
        klartraum::InterfaceCameraOrbit::UpDirection::Y);
    cameraOrbit->initialize(vulkanContext);
    cameraOrbit->setAzimuth(0.9f);
    cameraOrbit->setElevation(-0.5f);
    cameraOrbit->setPosition({-0.5f, 0.0f, 0.5f});
    cameraOrbit->setDistance(1.0f);
    engine.setInterfaceCamera(cameraOrbit);
    engine.setCameraUBO(cameraUBO);

    frontend.loop(maxFrames);

    if (maxFrames > 0) {
        std::cout << "\n--- GPU timing (mean over " << maxFrames << " frames) ---\n";
        for (auto& [name, ms] : engine.getProfilingResults()) {
            std::cout << "  " << name << ": " << ms << " ms\n";
        }
    }

    return 0;
}
