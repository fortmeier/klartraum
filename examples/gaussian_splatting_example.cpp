#include <cmath>
#include <iostream>
#include <filesystem>
#include <string>

#ifdef _WIN32
#include <windows.h>
#include <crtdbg.h>
#endif

#include "klartraum/glfw_frontend.hpp"
#include "klartraum/gaussian_splatting_factory.hpp"
#include "klartraum/gaussian_data_standard.hpp"
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
    //              --backend compute|raster   (select the rendering backend, default compute)
    //              --file PATH   (load a specific .spz scene)
    //              --camera-position X Y Z   (world-space eye position, looking at the origin)
    //              --flip-z   (mirror the loaded scene across the Z axis)
    int maxFrames = -1;
    klartraum::GsplatBackend backend = klartraum::GsplatBackend::Compute;
    std::string spzFile = "./3rdparty/spz/samples/racoonfamily.spz";
    glm::vec3 cameraPosition(0.55f, 0.48f, 0.69f);
    bool flipZ = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--frames" && i + 1 < argc) {
            try { maxFrames = std::stoi(argv[++i]); } catch (...) {}
        } else if (arg == "--backend" && i + 1 < argc) {
            std::string value = argv[++i];
            if (value == "compute") {
                backend = klartraum::GsplatBackend::Compute;
            } else if (value == "raster") {
                backend = klartraum::GsplatBackend::Raster;
            } else {
                std::cerr << "Unknown --backend value '" << value << "' (expected 'compute' or 'raster')" << std::endl;
                return 1;
            }
        } else if (arg == "--file" && i + 1 < argc) {
            spzFile = argv[++i];
        } else if (arg == "--camera-position") {
            if (i + 3 >= argc) {
                std::cerr << "--camera-position requires X Y Z" << std::endl;
                return 1;
            }
            try {
                cameraPosition.x = std::stof(argv[++i]);
                cameraPosition.y = std::stof(argv[++i]);
                cameraPosition.z = std::stof(argv[++i]);
            } catch (...) {
                std::cerr << "Invalid --camera-position (expected three numbers)" << std::endl;
                return 1;
            }
        } else if (arg == "--flip-z") {
            flipZ = true;
        }
    }

    const double cameraDistance = std::sqrt(
        cameraPosition.x * cameraPosition.x +
        cameraPosition.y * cameraPosition.y +
        cameraPosition.z * cameraPosition.z);
    if (!std::isfinite(cameraDistance) || cameraDistance <= 0.0) {
        std::cerr << "--camera-position must be finite and different from the origin" << std::endl;
        return 1;
    }

    std::cout << "Gaussian Splatting example";
    std::cout << " (backend: " << (backend == klartraum::GsplatBackend::Raster ? "raster" : "compute") << ")";
    if (maxFrames > 0) std::cout << " (closing after " << maxFrames << " frames)";
    std::cout << std::endl;
    std::cout << "Loading scene: " << spzFile << std::endl;
    if (flipZ) std::cout << "Flipping scene Z axis" << std::endl;
    std::cout << "Camera position: " << cameraPosition.x << " "
              << cameraPosition.y << " " << cameraPosition.z
              << " (looking at 0 0 0)" << std::endl;

    klartraum::GlfwFrontend frontend;
    auto& engine = frontend.getKlartraumEngine();
    auto& vulkanContext = engine.getVulkanContext();

    if (maxFrames > 0) {
        engine.enableProfiling();
        engine.enablePerformanceProfiling({"SM"});
    }

    // Loaded once; the graph builder below reuses it on every rebuild.
    auto model = std::make_shared<klartraum::GaussianDataStandard>(vulkanContext, spzFile, flipZ);

    // Everything tied to the swapchain is created in the graph builder, which
    // the engine runs now and again after each window resize.
    engine.setGraphBuilder([backend, model](klartraum::KlartraumEngine& e) {
        auto& vc = e.getVulkanContext();
        uint32_t numImages = vc.getNumberOfSwapChainImages();
        std::vector<VkImageView> imageViews(numImages);
        std::vector<VkImage>     images(numImages);
        std::vector<VkExtent2D>  extents(numImages, vc.getSwapChainExtent());
        for (uint32_t i = 0; i < numImages; ++i) {
            imageViews[i] = vc.getImageView(i);
            images[i]     = vc.getSwapChainImage(i);
        }
        auto imageViewSrc = std::make_shared<klartraum::ImageViewSrc>(imageViews, images, extents);
        for (uint32_t i = 0; i < numImages; ++i) {
            imageViewSrc->setWaitFor(i, vc.imageAvailableSemaphoresPerImage[i]);
        }

        auto cameraUBO = std::make_shared<klartraum::CameraUboType>();
        cameraUBO->setName("CameraUBO");

        auto splatting = klartraum::createGaussianSplatting(
            vc, backend, imageViewSrc, cameraUBO, model);
        e.add(splatting);
        e.setCameraUBO(cameraUBO);
    });

    auto cameraOrbit = std::make_shared<klartraum::InterfaceCameraOrbit>(
        klartraum::InterfaceCameraOrbit::UpDirection::Y);
    cameraOrbit->initialize(vulkanContext);
    cameraOrbit->setPosition({0.0f, 0.0f, 0.0f});
    cameraOrbit->setAzimuth(std::atan2(cameraPosition.z, cameraPosition.x));
    cameraOrbit->setElevation(-std::asin(cameraPosition.y / cameraDistance));
    cameraOrbit->setDistance(cameraDistance);
    engine.setInterfaceCamera(cameraOrbit);

    frontend.loop(maxFrames);

    if (maxFrames > 0) {
        std::cout << "\n--- GPU timing (mean over " << maxFrames << " frames) ---\n";
        for (auto& [name, ms] : engine.getProfilingResults()) {
            std::cout << "  " << name << ": " << ms << " ms\n";
        }
    }

    return 0;
}
