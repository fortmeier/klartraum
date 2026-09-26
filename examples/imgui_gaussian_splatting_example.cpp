#include <iostream>
#include <string>

#ifdef _WIN32
#include <windows.h>
#include <crtdbg.h>
#endif

#include <imgui.h>

#include "klartraum/imgui_frontend.hpp"
#include "klartraum/gaussian_splatting_factory.hpp"
#include "klartraum/gaussian_data_standard.hpp"
#include "klartraum/interface_camera_orbit.hpp"
#include "klartraum/computegraph/imageviewsrc.hpp"

namespace {

// Initial window size in screen coordinates; the default window is too small
// to leave room for the GUI next to the scene.
constexpr int kWindowWidth = 1024;
constexpr int kWindowHeight = 640;

const char* backendName(klartraum::GsplatBackend backend) {
    return backend == klartraum::GsplatBackend::Raster ? "raster" : "compute";
}

struct CameraDefaults {
    float azimuth = 0.9f;
    float elevation = -0.5f;
    glm::vec3 position = {-0.5f, 0.0f, 0.5f};
    float distance = 1.0f;
};

void resetCamera(klartraum::InterfaceCameraOrbit& camera) {
    const CameraDefaults defaults;
    camera.setAzimuth(defaults.azimuth);
    camera.setElevation(defaults.elevation);
    camera.setPosition(defaults.position);
    camera.setDistance(defaults.distance);
}

} // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportMode(_CRT_ERROR,  _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
    _CrtSetReportFile(_CRT_ERROR,  _CRTDBG_FILE_STDERR);
#endif

    // Parse args:  --frames N   (close after N frames)
    //              --spz PATH   (scene to load, default: raccoon sample)
    //              --backend compute|raster   (initial backend, default raster;
    //                                          switchable in the GUI)
    int maxFrames = -1;
    klartraum::GsplatBackend backend = klartraum::GsplatBackend::Raster;
    std::string spzFile = "./3rdparty/spz/samples/racoonfamily.spz";
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--frames" && i + 1 < argc) {
            try { maxFrames = std::stoi(argv[++i]); } catch (...) {}
        } else if (arg == "--spz" && i + 1 < argc) {
            spzFile = argv[++i];
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
        }
    }
    std::cout << "ImGui Gaussian Splatting example (backend: " << backendName(backend) << ")";
    if (maxFrames > 0) std::cout << " (closing after " << maxFrames << " frames)";
    std::cout << std::endl;

    klartraum::ImGuiFrontend frontend;
    auto& engine = frontend.getKlartraumEngine();
    auto& vulkanContext = engine.getVulkanContext();

    // Loaded once; the graph builder below reuses it on every rebuild.
    auto model = std::make_shared<klartraum::GaussianDataStandard>(vulkanContext, spzFile);

    // Changing the config needs new pipelines, so the GUI edits `pendingConfig`
    // and the graphs are rebuilt with it (see "Apply" below).
    klartraum::GsplatConfig config;
    klartraum::GsplatConfig pendingConfig = config;

    // Everything tied to the swapchain is created in the graph builder, which
    // the engine runs now and again after each window resize.
    auto graphBuilder = [&backend, &config, model](klartraum::KlartraumEngine& e) {
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
            vc, backend, imageViewSrc, cameraUBO, model, config);
        e.add(splatting);
        e.setCameraUBO(cameraUBO);
    };
    engine.setGraphBuilder(graphBuilder);

    auto cameraOrbit = std::make_shared<klartraum::InterfaceCameraOrbit>(
        klartraum::InterfaceCameraOrbit::UpDirection::Y);
    cameraOrbit->initialize(vulkanContext);
    resetCamera(*cameraOrbit);
    engine.setInterfaceCamera(cameraOrbit);

    // The resize is picked up by the first frame's event processing, which
    // rebuilds the graphs for the larger swapchain.
    glfwSetWindowSize(frontend.getGlfwWindow(), kWindowWidth, kWindowHeight);

    bool showDemoWindow = false;

    // The GUI runs between frames; once the GPU is idle the old graphs can be
    // released and rebuilt (new backend or config).
    auto rebuildGraphs = [&]() {
        vkDeviceWaitIdle(vulkanContext.getDevice());
        engine.setGraphBuilder(graphBuilder);
    };

    frontend.setGui([&]() {
        const ImGuiIO& io = ImGui::GetIO();

        ImGui::SetNextWindowPos(ImVec2(10.0f, 10.0f), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(320.0f, 0.0f), ImGuiCond_FirstUseEver);
        ImGui::Begin("Gaussian Splatting");

        if (ImGui::CollapsingHeader("Statistics", ImGuiTreeNodeFlags_DefaultOpen)) {
            const VkExtent2D extent = vulkanContext.getSwapChainExtent();
            ImGui::Text("%.1f FPS (%.2f ms)", io.Framerate, 1000.0f / io.Framerate);
            ImGui::Text("Resolution: %u x %u", extent.width, extent.height);
            ImGui::Text("Gaussians: %u", model->count());
            ImGui::Text("Backend: %s", backendName(backend));
        }

        if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
            // The camera also follows mouse input, so show its current state
            // and write back only what the user edits here.
            float azimuth = static_cast<float>(cameraOrbit->getAzimuth());
            float elevation = static_cast<float>(cameraOrbit->getElevation());
            float distance = static_cast<float>(cameraOrbit->getDistance());
            glm::vec3 position = cameraOrbit->getPosition();

            if (ImGui::SliderFloat("Azimuth", &azimuth, -3.1416f, 3.1416f)) {
                cameraOrbit->setAzimuth(azimuth);
            }
            if (ImGui::SliderFloat("Elevation", &elevation, -1.5f, 1.5f)) {
                cameraOrbit->setElevation(elevation);
            }
            if (ImGui::SliderFloat("Distance", &distance, 0.05f, 10.0f, "%.2f", ImGuiSliderFlags_Logarithmic)) {
                cameraOrbit->setDistance(distance);
            }
            if (ImGui::DragFloat3("Target", &position.x, 0.01f)) {
                cameraOrbit->setPosition(position);
            }
            if (ImGui::Button("Reset camera")) {
                resetCamera(*cameraOrbit);
            }
        }

        if (ImGui::CollapsingHeader("Rendering", ImGuiTreeNodeFlags_DefaultOpen)) {
            // Switching the backend takes effect right away.
            const klartraum::GsplatBackend backends[] = {
                klartraum::GsplatBackend::Raster, klartraum::GsplatBackend::Compute};
            if (ImGui::BeginCombo("Backend", backendName(backend))) {
                for (auto candidate : backends) {
                    const bool selected = candidate == backend;
                    if (ImGui::Selectable(backendName(candidate), selected) && !selected) {
                        backend = candidate;
                        rebuildGraphs();
                    }
                    if (selected) {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }

            // Each backend reads only some GsplatConfig fields; show those.
            bool changed = false;
            if (backend == klartraum::GsplatBackend::Raster) {
                ImGui::SliderInt("SH degree", &pendingConfig.shDegree, 0, 3);
                ImGui::SliderFloat("Alpha cull", &pendingConfig.alphaCullThreshold, 0.0f, 0.2f, "%.4f",
                                   ImGuiSliderFlags_Logarithmic);

                ImGui::BeginDisabled(!vulkanContext.isMeshShaderSupported());
                ImGui::Checkbox("Mesh shader path", &pendingConfig.useMeshShader);
                ImGui::EndDisabled();

                changed = pendingConfig.shDegree != config.shDegree
                       || pendingConfig.alphaCullThreshold != config.alphaCullThreshold
                       || pendingConfig.useMeshShader != config.useMeshShader;
            } else {
                ImGui::SliderFloat("Spread", &pendingConfig.spreadMultiplier, 1.0f, 4.0f, "%.2f sigma");

                changed = pendingConfig.spreadMultiplier != config.spreadMultiplier;
            }

            ImGui::BeginDisabled(!changed);
            if (ImGui::Button("Apply")) {
                config = pendingConfig;
                rebuildGraphs();
            }
            ImGui::EndDisabled();
        }

        ImGui::Separator();
        ImGui::Checkbox("Show ImGui demo window", &showDemoWindow);
        ImGui::TextDisabled("Left drag: orbit, wheel: zoom");

        ImGui::End();

        if (showDemoWindow) {
            ImGui::ShowDemoWindow(&showDemoWindow);
        }
    });

    frontend.loop(maxFrames);

    return 0;
}
