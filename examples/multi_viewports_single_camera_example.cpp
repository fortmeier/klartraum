// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Renders the same scene twice, side by side, in one window: the raster
// backend in the left half, the compute backend in the right half. Both read
// one camera uniform buffer, so a single orbit camera moves both views.
//
// The window has one swapchain. Window::makeViewport(x, y, w, h) returns a
// render target (an ImageViewSrc) backed by its own offscreen image; the
// engine composites all viewports into their rectangles of the swapchain image
// and presents once per frame.

#include <iostream>
#include <string>

#include "klartraum/glfw_frontend.hpp"
#include "klartraum/gaussian_splatting_factory.hpp"
#include "klartraum/gaussian_data_standard.hpp"
#include "klartraum/interface_camera_orbit.hpp"

int main(int /*argc*/, char** /*argv*/) {
    klartraum::GlfwFrontend frontend;
    auto& engine = frontend.getKlartraumEngine();
    auto& vulkanContext = engine.getVulkanContext();

    // ---- The single shared camera -----------------------------------------
    // One view and projection, shared by both equally sized viewports.
    auto cameraUBO = std::make_shared<klartraum::CameraUboType>();
    cameraUBO->setName("CameraUBO");

    // ---- Two viewport targets carved out of the one swapchain --------------
    // makeViewport(x, y, w, h) returns the same kind of render target the scene
    // factories already accept (an ImageViewSrc). Each is backed by its own
    // offscreen image sized to the viewport; the engine's per-frame composite
    // blits both into their (x, y, w, h) rectangles of the one swapchain image
    // and presents once. No manual VkImageView / VkImage / semaphore plumbing.
    auto extent = vulkanContext.getSwapChainExtent();
    uint32_t halfW = extent.width / 2;

    auto leftViewport = engine.getWindow().makeViewport(0, 0, halfW, 0 + extent.height);
    auto rightViewport = engine.getWindow().makeViewport(halfW, 0, extent.width - halfW, extent.height);

    // ---- Two scenes, same camera, different backends, different targets ----
    // The lantern capture is stored with Y pointing down, hence flipY.
    std::string spzFile = "./data/lantern.spz";

    // One model, loaded/uploaded once, shared by both backends.
    auto model = std::make_shared<klartraum::GaussianDataStandard>(vulkanContext, spzFile, true);

    auto splatCompute = klartraum::createGaussianSplatting(vulkanContext, klartraum::GsplatBackend::Compute,
                                                           rightViewport, cameraUBO, model);

    auto splatRaster = klartraum::createGaussianSplatting(vulkanContext, klartraum::GsplatBackend::Raster, leftViewport,
                                                          cameraUBO, model);

    // Both graphs go to the engine.  Because both viewport targets came from
    // the engine's single swapchain, the engine is responsible for sequencing
    // them so both have written their sub-region before the one present.
    engine.add(splatCompute);
    engine.add(splatRaster);

    // ---- One camera drives the shared UBO ----------------------------------
    auto cameraOrbit =
        std::make_shared<klartraum::InterfaceCameraOrbit>(klartraum::InterfaceCameraOrbit::UpDirection::Y);
    cameraOrbit->initialize(vulkanContext);
    auto viewportExtent = leftViewport->getImageExtent(0);
    cameraOrbit->setProjectionAspectRatio(viewportExtent.width / static_cast<float>(viewportExtent.height));
    cameraOrbit->setAzimuth(0.9f);
    cameraOrbit->setElevation(-0.5f);
    cameraOrbit->setPosition({0.0f, -0.08f, 0.0f});
    cameraOrbit->setDistance(0.6f);
    engine.setInterfaceCamera(cameraOrbit);
    engine.setCameraUBO(cameraUBO);

    frontend.loop();
    return 0;
}
