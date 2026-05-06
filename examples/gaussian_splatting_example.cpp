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

    // Parse --frames N  (optional: close after N rendered frames)
    int maxFrames = -1;
    for (int i = 1; i < argc - 1; ++i) {
        if (std::string(argv[i]) == "--frames") {
            try { maxFrames = std::stoi(argv[i + 1]); } catch (...) {}
        }
    }
    if (maxFrames > 0)
        std::cout << "Gaussian Splatting example (closing after " << maxFrames << " frames)" << std::endl;
    else
        std::cout << "Gaussian Splatting example" << std::endl;

    klartraum::GlfwFrontend frontend;
    auto& engine = frontend.getKlartraumEngine();
    auto& vulkanContext = engine.getVulkanContext();

    // Build an ImageViewSrc from the real swapchain images so
    // VulkanGaussianSplatting can write directly to the presentable images.
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

    // Camera UBO — used by both the projection shader and the camera orbit.
    auto cameraUBO = std::make_shared<klartraum::CameraUboType>();
    cameraUBO->setName("CameraUBO");

    std::string spzFile = "./3rdparty/spz/samples/racoonfamily.spz";
    std::shared_ptr<klartraum::VulkanGaussianSplatting> splatting =
        vulkanContext.create<klartraum::VulkanGaussianSplatting>(imageViewSrc, cameraUBO, spzFile);

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

    return 0;
}
