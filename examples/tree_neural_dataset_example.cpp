// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Renders a database of view pairs of the bonsai landscape (scale 1) for training a network that enhances the
// Gaussian renderer's distant trees:
//
//   tree_neural_dataset_example [--views 4] [--first 0] [--width 1280] [--height 720] [--supersample 2]
//                               [--batch 256] [--range 1500] [--season 2] [--seed 1]
//                               [--out-dir build/TestingOutput/neural_dataset]
//
// For every view: the Gaussian BonsaiRenderer's image as it runs (low fidelity), and the same view by
// TreeReferenceRenderer (every far tree generated in full by the classical renderer) at --supersample times the
// resolution, box-filtered down (high fidelity), with the mask of the far trees' pixels (the share of the
// supersampled pixels they cover). The views are drawn from --seed: at eye height on open land looking towards
// forest, and 30, 120 and 400 m above the soil looking down onto forest. --first skips views, to continue a database.
//
// Writes <out-dir>/lr_NNNN.ppm, hf_NNNN.ppm, mask_NNNN.pgm and appends to <out-dir>/views.csv (pose, far trees,
// batches, seconds per rendering).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <glm/gtc/matrix_transform.hpp>

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/offscreen_target.hpp"
#include "klartraum/tree/bonsai_renderer.hpp"
#include "klartraum/tree/bonsai_scene.hpp"
#include "klartraum/tree/tree_reference.hpp"

using namespace klartraum;

namespace {

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

void writePgm(const std::filesystem::path& path, const std::vector<uint8_t>& gray, VkExtent2D extent) {
    std::ofstream file(path, std::ios::binary);
    file << "P5\n" << extent.width << " " << extent.height << "\n255\n";
    file.write(reinterpret_cast<const char*>(gray.data()), std::streamsize(gray.size()));
}

// The box filter of a BGRA image by `factor`, and of its one-channel mask.
std::vector<uint8_t> boxDown(const std::vector<uint8_t>& image, VkExtent2D from, uint32_t factor, uint32_t channels) {
    const uint32_t w = from.width / factor, h = from.height / factor;
    std::vector<uint8_t> out(size_t(w) * h * channels);
    for (uint32_t y = 0; y < h; ++y) {
        for (uint32_t x = 0; x < w; ++x) {
            for (uint32_t c = 0; c < channels; ++c) {
                uint32_t sum = 0;
                for (uint32_t j = 0; j < factor; ++j) {
                    for (uint32_t i = 0; i < factor; ++i) {
                        sum += image[((size_t(y) * factor + j) * from.width + x * factor + i) * channels + c];
                    }
                }
                out[(size_t(y) * w + x) * channels + c] = uint8_t((sum + factor * factor / 2) / (factor * factor));
            }
        }
    }
    return out;
}

struct Pose {
    glm::vec3 eye;
    glm::vec3 lookAt;
    float nearPlane;
    int kind; // 0 at eye height on open land, 1-3: 30, 120, 400 m above the soil
};

// View `index` of the database: a place over the pot's landscape with forest around, by kind.
Pose poseOf(const BonsaiScene& scene, uint32_t seed, uint32_t index) {
    std::mt19937 rng(seed * 7919u + index * 104729u);
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);
    const float m = 1.0f / scene.config.scale.metersPerUnit();
    const TreeTerrainParams& t = scene.terrain;
    const int kind = int(index % 4u);
    const float above[4] = {1.7f, 30.0f, 120.0f, 400.0f};
    const float pitchDegrees[4] = {1.0f, -6.0f, -14.0f, -32.0f};
    for (int attempt = 0;; ++attempt) {
        const float x = (2.0f * unit(rng) - 1.0f) * t.outline.x;
        const float z = (2.0f * unit(rng) - 1.0f) * t.outline.y;
        if (tree::treeTerrainRadius(t, x, z) > 0.8f) {
            continue;
        }
        // Forest ahead (200 m ahead at eye height, where the camera stands on open land; 3 times the height
        // ahead above it).
        const float heading = 2.0f * glm::pi<float>() * unit(rng);
        const glm::vec3 ahead(std::sin(heading), 0.0f, -std::cos(heading));
        const glm::vec3 probe = glm::vec3(x, 0.0f, z) + (kind == 0 ? 200.0f : 3.0f * above[kind]) * m * ahead;
        const bool open = tree::treeLandCover(t, x, z).vegetation.x < 0.2f;
        if ((tree::treeLandCover(t, probe.x, probe.z).vegetation.x < 0.6f || (kind == 0 && !open)) && attempt < 2000) {
            continue;
        }
        Pose pose;
        pose.kind = kind;
        pose.eye = glm::vec3(x, scene.soilHeight(x, z) + above[kind] * m, z);
        const float pitch = glm::radians(pitchDegrees[kind]);
        pose.lookAt = pose.eye + glm::vec3(ahead.x * std::cos(pitch), std::sin(pitch), ahead.z * std::cos(pitch));
        pose.nearPlane = std::clamp(0.3f * above[kind], 0.05f, 5.0f) * m;
        return pose;
    }
}

} // namespace

int main(int argc, char** argv) {
    uint32_t views = 4, first = 0, width = 1280, height = 720, supersample = 2, batch = 256, seed = 1;
    float range = 1500.0f, season = 2.0f;
    std::filesystem::path outDir = "build/TestingOutput/neural_dataset";
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const bool hasValue = i + 1 < argc;
        if (arg == "--views" && hasValue) {
            views = uint32_t(std::stoul(argv[++i]));
        } else if (arg == "--first" && hasValue) {
            first = uint32_t(std::stoul(argv[++i]));
        } else if (arg == "--width" && hasValue) {
            width = uint32_t(std::stoul(argv[++i]));
        } else if (arg == "--height" && hasValue) {
            height = uint32_t(std::stoul(argv[++i]));
        } else if (arg == "--supersample" && hasValue) {
            supersample = std::max(1u, uint32_t(std::stoul(argv[++i])));
        } else if (arg == "--batch" && hasValue) {
            batch = uint32_t(std::stoul(argv[++i]));
        } else if (arg == "--range" && hasValue) {
            range = std::stof(argv[++i]);
        } else if (arg == "--season" && hasValue) {
            season = std::stof(argv[++i]);
        } else if (arg == "--seed" && hasValue) {
            seed = uint32_t(std::stoul(argv[++i]));
        } else if (arg == "--out-dir" && hasValue) {
            outDir = argv[++i];
        } else {
            std::cerr << "unknown option " << arg << std::endl;
            return 1;
        }
    }

    BonsaiSceneConfig sceneConfig;
    sceneConfig.scale.s = 1.0f;
    sceneConfig.seed = 7;
    const BonsaiScene scene = makeBonsaiScene(sceneConfig);
    HeadlessFrontend frontend;
    VulkanContext& vc = frontend.getKlartraumEngine().getVulkanContext();

    // The Gaussian renderer as the bonsai explorer runs it.
    const VkExtent2D extent{width, height};
    auto lowTarget = std::make_shared<OffscreenTarget>(vc, extent, 1);
    BonsaiRenderConfig lowConfig;
    lowConfig.extent = extent;
    lowConfig.mode = BonsaiRenderMode::Gaussian;
    BonsaiRenderer low(vc, scene, lowTarget, lowConfig);
    ComputeGraph lowGraph(vc, 1);
    lowGraph.compileFrom(low.getOutput());
    low.initialize(1);
    low.setSeason(season);

    TreeReferenceConfig referenceConfig;
    referenceConfig.batch = batch;
    referenceConfig.rangeMeters = range;
    const VkExtent2D fine{width * supersample, height * supersample};
    TreeReferenceRenderer high(vc, scene, fine, referenceConfig);

    std::filesystem::create_directories(outDir);
    const bool fresh = !std::filesystem::exists(outDir / "views.csv");
    std::ofstream csv(outDir / "views.csv", std::ios::app);
    if (fresh) {
        csv << "view,kind,eye_x,eye_y,eye_z,look_x,look_y,look_z,near,far_trees,batches,low_s,high_s\n";
    }
    const float fov = 50.0f, farPlane = 50.0f;
    for (uint32_t v = first; v < first + views; ++v) {
        const Pose pose = poseOf(scene, seed, v);

        const auto lowStart = std::chrono::steady_clock::now();
        CameraMVP& ubo = low.getCamera()->ubo;
        ubo.model = glm::mat4(1.0f);
        ubo.view = glm::lookAt(pose.eye, pose.lookAt, glm::vec3(0.0f, 1.0f, 0.0f));
        ubo.proj = glm::perspective(glm::radians(fov), float(width) / float(height), pose.nearPlane, farPlane);
        ubo.proj[1][1] *= -1.0f;
        ubo.cameraWorldPos = glm::vec4(pose.eye, 1.0f);
        low.getCamera()->update(0);
        low.setTime(0.0f, 0.5f, season, glm::vec4(1.0f, 0.0f, 0.0f, 0.0f));
        for (int i = 0; i < 40; ++i) {
            low.setView(pose.eye);
            lowGraph.submitAndWait(vc.getGraphicsQueue(), 0);
            if (low.isSettled()) {
                break;
            }
        }
        // As the explorer runs it: the copies and the canopy follow the last frame's trees.
        for (int i = 0; i < 3; ++i) {
            low.setView(pose.eye);
            lowGraph.submitAndWait(vc.getGraphicsQueue(), 0);
        }
        const std::vector<uint8_t> lowImage = readImage(vc, lowTarget->getImage(0), extent);
        const double lowSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - lowStart).count();

        const auto highStart = std::chrono::steady_clock::now();
        const TreeReferenceRenderer::Frame frame =
            high.render(pose.eye, pose.lookAt, fov, pose.nearPlane, farPlane, season, 0.5f);
        const double highSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - highStart).count();

        char name[32];
        std::snprintf(name, sizeof(name), "%04u", v);
        writePpm(outDir / (std::string("lr_") + name + ".ppm"), lowImage, extent);
        writePpm(outDir / (std::string("hf_") + name + ".ppm"), boxDown(frame.image, fine, supersample, 4), extent);
        writePgm(outDir / (std::string("mask_") + name + ".pgm"), boxDown(frame.mask, fine, supersample, 1), extent);
        csv << v << "," << pose.kind << "," << pose.eye.x << "," << pose.eye.y << "," << pose.eye.z << ","
            << pose.lookAt.x << "," << pose.lookAt.y << "," << pose.lookAt.z << "," << pose.nearPlane << ","
            << frame.trees << "," << frame.batches << "," << lowSeconds << "," << highSeconds << std::endl;
        std::printf("view %u (kind %d): %u far trees in %u batches; low %.1f s, high %.1f s\n", v, pose.kind,
                    frame.trees, frame.batches, lowSeconds, highSeconds);
    }
    return 0;
}
