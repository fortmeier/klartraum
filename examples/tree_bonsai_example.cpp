// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Renders the bonsai forest (Kuth et al., Real-Time GPU Tree Generation, HPG
// 2025) headlessly from cameras around it, classically and as Gaussians, and
// writes one PPM image per view.
//
//   tree_bonsai_example [--renderer classical|gaussian|both] [--backend raster|compute]
//                       [--views 4] [--trees 13] [--seed 7] [--width 1280] [--height 720]
//                       [--distance 4.2] [--elevation 0.32] [--azimuth 0.35]
//                       [--out-dir build/TestingOutput/tree_bonsai]
//
// The trees are generated on the GPU every frame; per view the CPU only sets
// the camera. View i is seen from azimuth start + 2*pi*i/views. Images go to
// <out-dir>/frames/<renderer>_<view>.ppm.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <glm/gtc/matrix_transform.hpp>

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/offscreen_target.hpp"
#include "klartraum/tree/bonsai_renderer.hpp"

using namespace klartraum;

namespace {

// Reads an OffscreenTarget image back as tightly packed BGRA bytes; the renderers leave it in TRANSFER_SRC_OPTIMAL.
std::vector<uint8_t> readImage(VulkanContext& vc, VkImage image, VkExtent2D extent) {
    const VkDeviceSize bytes = VkDeviceSize(extent.width) * extent.height * 4;
    VkBuffer buffer;
    VkDeviceMemory memory;
    vc.createBuffer(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, buffer, memory);
    vc.submitImmediate([&](VkCommandBuffer commandBuffer) {
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {extent.width, extent.height, 1};
        vkCmdCopyImageToBuffer(commandBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer, 1, &region);
    });
    void* data;
    vkMapMemory(vc.getDevice(), memory, 0, bytes, 0, &data);
    std::vector<uint8_t> pixels(static_cast<uint8_t*>(data), static_cast<uint8_t*>(data) + bytes);
    vkUnmapMemory(vc.getDevice(), memory);
    vkDestroyBuffer(vc.getDevice(), buffer, nullptr);
    vkFreeMemory(vc.getDevice(), memory, nullptr);
    return pixels;
}

void writePpm(const std::filesystem::path& path, const std::vector<uint8_t>& bgra, VkExtent2D extent) {
    std::ofstream file(path, std::ios::binary);
    file << "P6\n" << extent.width << " " << extent.height << "\n255\n";
    for (size_t i = 0; i < bgra.size(); i += 4) {
        const char rgb[3] = {char(bgra[i + 2]), char(bgra[i + 1]), char(bgra[i])};
        file.write(rgb, 3);
    }
}

void setCamera(CameraMVP& ubo, glm::vec3 eye, glm::vec3 target, float aspect) {
    ubo.model = glm::mat4(1.0f);
    ubo.view = glm::lookAt(eye, target, glm::vec3(0.0f, 1.0f, 0.0f));
    ubo.proj = glm::perspective(glm::radians(40.0f), aspect, 0.05f, 50.0f);
    ubo.proj[1][1] *= -1.0f;
    ubo.cameraWorldPos = glm::vec4(eye, 1.0f);
}

} // namespace

int main(int argc, char** argv) {
    std::string renderer = "both";
    std::string backendName = "raster";
    uint32_t views = 4;
    BonsaiSceneConfig sceneConfig;
    VkExtent2D extent{1280, 720};
    float distance = 4.2f;
    float elevation = 0.32f;
    float azimuth = 0.35f;
    std::filesystem::path outDir = "build/TestingOutput/tree_bonsai";

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const bool hasValue = i + 1 < argc;
        if (arg == "--renderer" && hasValue) {
            renderer = argv[++i];
        } else if (arg == "--backend" && hasValue) {
            backendName = argv[++i];
        } else if (arg == "--views" && hasValue) {
            views = uint32_t(std::stoul(argv[++i]));
        } else if (arg == "--trees" && hasValue) {
            sceneConfig.treeCount = uint32_t(std::stoul(argv[++i]));
        } else if (arg == "--seed" && hasValue) {
            sceneConfig.seed = uint32_t(std::stoul(argv[++i]));
        } else if (arg == "--width" && hasValue) {
            extent.width = uint32_t(std::stoul(argv[++i]));
        } else if (arg == "--height" && hasValue) {
            extent.height = uint32_t(std::stoul(argv[++i]));
        } else if (arg == "--distance" && hasValue) {
            distance = std::stof(argv[++i]);
        } else if (arg == "--elevation" && hasValue) {
            elevation = std::stof(argv[++i]);
        } else if (arg == "--azimuth" && hasValue) {
            azimuth = std::stof(argv[++i]);
        } else if (arg == "--out-dir" && hasValue) {
            outDir = argv[++i];
        } else {
            std::cerr << "unknown or incomplete option " << arg << std::endl;
            return 1;
        }
    }
    if (renderer != "classical" && renderer != "gaussian" && renderer != "both") {
        std::cerr << "--renderer must be classical, gaussian or both" << std::endl;
        return 1;
    }
    if (backendName != "raster" && backendName != "compute") {
        std::cerr << "--backend must be raster or compute" << std::endl;
        return 1;
    }

    HeadlessFrontend frontend;
    VulkanContext& vc = frontend.getKlartraumEngine().getVulkanContext();
    const BonsaiScene scene = makeBonsaiScene(sceneConfig);
    std::filesystem::create_directories(outDir / "frames");
    std::cout << scene.trees.size() << " trees" << std::endl;

    std::vector<BonsaiRenderMode> modes;
    if (renderer != "gaussian") {
        modes.push_back(BonsaiRenderMode::Classical);
    }
    if (renderer != "classical") {
        modes.push_back(BonsaiRenderMode::Gaussian);
    }
    for (BonsaiRenderMode mode : modes) {
        const char* modeName = mode == BonsaiRenderMode::Classical ? "classical" : "gaussian";
        BonsaiRenderConfig config;
        config.extent = extent;
        config.mode = mode;
        config.backend = backendName == "raster" ? GsplatBackend::Raster : GsplatBackend::Compute;

        auto target = std::make_shared<OffscreenTarget>(vc, extent, 1);
        BonsaiRenderer bonsai(vc, scene, target, config);
        ComputeGraph graph(vc, 1);
        graph.compileFrom(bonsai.getOutput());
        bonsai.initialize(1);

        for (uint32_t view = 0; view < views; ++view) {
            const float angle = azimuth + 2.0f * 3.14159265f * float(view) / float(views);
            const glm::vec3 eye =
                scene.focus + distance * glm::vec3(std::sin(angle) * std::cos(elevation), std::sin(elevation),
                                                   std::cos(angle) * std::cos(elevation));
            setCamera(bonsai.getCamera()->ubo, eye, scene.focus, float(extent.width) / float(extent.height));
            bonsai.getCamera()->update(0);
            const auto start = std::chrono::steady_clock::now();
            graph.submitAndWait(vc.getGraphicsQueue(), 0);
            const double ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            char name[64];
            std::snprintf(name, sizeof(name), "%s_%03u.ppm", modeName, view);
            writePpm(outDir / "frames" / name, readImage(vc, target->getImage(0), extent), extent);
            std::cout << modeName << " view " << view << ": " << ms << " ms" << std::endl;
        }
    }
    return 0;
}
