# A first application

This walks through a shortened version of
[`examples/gaussian_splatting_example.cpp`](https://github.com/fortmeier/klartraum/blob/main/examples/gaussian_splatting_example.cpp),
which opens a window and renders a Gaussian splatting scene with an orbit
camera.

```cpp
#include "klartraum/glfw_frontend.hpp"
#include "klartraum/gaussian_splatting_factory.hpp"
#include "klartraum/gaussian_data_standard.hpp"
#include "klartraum/interface_camera_orbit.hpp"
#include "klartraum/computegraph/imageviewsrc.hpp"

int main() {
    klartraum::GlfwFrontend frontend;
    auto& engine = frontend.getKlartraumEngine();
    auto& vulkanContext = engine.getVulkanContext();

    // Load the scene once; the graph builder below reuses it. The lantern
    // capture is stored with Y pointing down, so it is loaded with flipY.
    auto model = std::make_shared<klartraum::GaussianDataStandard>(
        vulkanContext, "./data/lantern.spz", true);

    // The graph builder creates everything tied to the swapchain. The engine
    // runs it now and again after every window resize.
    engine.setGraphBuilder([model](klartraum::KlartraumEngine& e) {
        auto& vc = e.getVulkanContext();
        uint32_t numImages = vc.getNumberOfSwapChainImages();
        std::vector<VkImageView> imageViews(numImages);
        std::vector<VkImage>     images(numImages);
        std::vector<VkExtent2D>  extents(numImages, vc.getSwapChainExtent());
        for (uint32_t i = 0; i < numImages; ++i) {
            imageViews[i] = vc.getImageView(i);
            images[i]     = vc.getSwapChainImage(i);
        }
        auto target = std::make_shared<klartraum::ImageViewSrc>(imageViews, images, extents);
        for (uint32_t i = 0; i < numImages; ++i) {
            target->setWaitFor(i, vc.imageAvailableSemaphoresPerImage[i]);
        }

        auto cameraUBO = std::make_shared<klartraum::CameraUboType>();
        auto splatting = klartraum::createGaussianSplatting(
            vc, klartraum::GsplatBackend::Compute, target, cameraUBO, model);

        e.add(splatting);
        e.setCameraUBO(cameraUBO);
    });

    auto camera = std::make_shared<klartraum::InterfaceCameraOrbit>(
        klartraum::InterfaceCameraOrbit::UpDirection::Y);
    camera->initialize(vulkanContext);
    camera->setPosition({0.0f, -0.08f, 0.0f});
    camera->setDistance(0.6);
    engine.setInterfaceCamera(camera);

    frontend.loop();
    return 0;
}
```

What happens here:

1. {cpp:class}`klartraum::GlfwFrontend` creates a window, the Vulkan context
   and the engine.
2. {cpp:class}`klartraum::GaussianDataStandard` loads an `.spz` file into GPU
   buffers.
3. The graph builder wires a small compute graph: the swapchain images are the
   render target ({cpp:class}`klartraum::ImageViewSrc`), and
   `createGaussianSplatting` adds the splatting pipeline that draws into them.
   There is one execution path per swapchain image (see
   {doc}`../concepts/compute-graph`).
4. `frontend.loop()` runs the frame loop: it updates the camera, submits the
   pre-recorded command buffers and presents the result.
