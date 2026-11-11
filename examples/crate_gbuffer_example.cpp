// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// A toy scene for the photo enhancer pipeline: a crate (a box whose six faces have their own surface coordinates and
// object ids) standing on a floor, rendered by GBufferMeshRenderer from random cameras with all G-buffers. The views
// are written like the frames of tree_bonsai_example --fly-in --gbuffers (frames/classical_*.png and
// cameras_classical.csv), so scripts/neural/prepare_photo_pairs.py and add_surface_coordinates.py take them.
//
// With --orbit the camera circles the crate instead (a turntable), and with --model every view is also enhanced by a
// photo enhancer trained on the crate (neural::PhotoEnhancer: the learnable texture and the generator, on the GPU)
// into frames/enhanced_NNNN.png, under the appearance code of a training image (--appearance) or their mean.
//
//   crate_gbuffer_example --views 180 --size 512 --seed 1 --out-dir build/TestingOutput/crate/flights/crate_a
//   crate_gbuffer_example --orbit --views 240 --model build/TestingOutput/crate/runs/c1/photo_enhancer_best

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
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
#include "klartraum/gbuffer_mesh_renderer.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/image_io.hpp"
#include "klartraum/neural/photo_enhancer.hpp"
#include "klartraum/offscreen_target.hpp"
#include "klartraum/tree/tree_model.hpp"

using namespace klartraum;

namespace {

constexpr float kFovDegrees = 40.0f;
constexpr float kDepthFar = 60.0f; // the depth window, beyond the floor's far corners

// Reads an OffscreenTarget image back as tightly packed BGRA bytes; the render pass leaves it in TRANSFER_SRC_OPTIMAL.
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

} // namespace

int main(int argc, char** argv) {
    std::filesystem::path outDir = "build/TestingOutput/crate/flights/crate_a";
    uint32_t views = 180, size = 512, seed = 1, sampleSize = 256;
    bool orbit = false;
    std::string model;
    int appearanceRow = -1; // -1: the training codes' mean
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const bool hasValue = i + 1 < argc;
        if (arg == "--out-dir" && hasValue)
            outDir = argv[++i];
        else if (arg == "--views" && hasValue)
            views = uint32_t(std::stoul(argv[++i]));
        else if (arg == "--size" && hasValue)
            size = uint32_t(std::stoul(argv[++i]));
        else if (arg == "--seed" && hasValue)
            seed = uint32_t(std::stoul(argv[++i]));
        else if (arg == "--orbit")
            orbit = true;
        else if (arg == "--model" && hasValue)
            model = argv[++i];
        else if (arg == "--appearance" && hasValue)
            appearanceRow = std::stoi(argv[++i]);
        else if (arg == "--sample-size" && hasValue)
            sampleSize = uint32_t(std::stoul(argv[++i]));
        else {
            std::cerr << "unknown argument " << arg << std::endl;
            return 1;
        }
    }
    try {
        HeadlessFrontend frontend;
        VulkanContext& vc = frontend.getKlartraumEngine().getVulkanContext();
        // The crate's faces get ids spread over the 16 bits, (face + 0.5) / 6 of the range, so that the texture's id
        // axis holds six well separated slices; the floor sits in between at 1/2.
        std::array<uint32_t, 6> faceIds{};
        for (uint32_t face = 0; face < 6; ++face)
            faceIds[face] = uint32_t(std::lround((float(face) + 0.5f) / 6.0f * 65535.0f));
        auto vertices = gbufferBox(glm::vec3(-0.5f, 0.0f, -0.5f), glm::vec3(0.5f, 1.0f, 0.5f), tree::TREE_CLASS_STEM,
                                   glm::vec3(0.62f, 0.45f, 0.28f), faceIds);
        const auto floor = gbufferFloor(40.0f, 0.0f, 4.0f, tree::TREE_CLASS_SOIL, glm::vec3(0.5f, 0.5f, 0.48f), 32768u);
        vertices.insert(vertices.end(), floor.begin(), floor.end());
        const VkExtent2D extent{size, size};
        auto target = std::make_shared<OffscreenTarget>(vc, extent, 1);
        GBufferMeshRenderer renderer(vc, vertices, target, extent);
        ComputeGraph graph(vc, 1);
        graph.compileFrom(renderer.getOutput());
        std::unique_ptr<neural::PhotoEnhancer> enhancer;
        if (!model.empty()) {
            enhancer = std::make_unique<neural::PhotoEnhancer>(vc, model, sampleSize, sampleSize);
            if (appearanceRow >= 0) {
                const uint32_t n = enhancer->appearanceSize();
                const auto& codes = enhancer->trainingCodes();
                if (n == 0 || size_t(appearanceRow + 1) * n > codes.size())
                    throw std::runtime_error("--appearance: the model has no such training code");
                enhancer->setAppearance(
                    std::vector<float>(codes.begin() + appearanceRow * n, codes.begin() + (appearanceRow + 1) * n));
                std::cout << "appearance of " << enhancer->trainingIds()[size_t(appearanceRow)] << std::endl;
            }
        }

        std::filesystem::create_directories(outDir / "frames");
        std::ofstream cameras(outDir / "cameras_classical.csv");
        cameras << "frame,eye_x,eye_y,eye_z,look_x,look_y,look_z,near_plane,fov_y_degrees,depth_near,depth_far\n";
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
        std::normal_distribution<float> jitter(0.0f, 0.12f);
        const std::pair<const char*, GBufferOutput> outputs[] = {
            {"", GBufferOutput::Colour},         {"class_", GBufferOutput::Classes},
            {"depth_", GBufferOutput::Depth},    {"depth16_", GBufferOutput::Depth16},
            {"u_", GBufferOutput::CoordinatesU}, {"v_", GBufferOutput::CoordinatesV}};
        for (uint32_t view = 0; view < views; ++view) {
            // Any azimuth, 8 to 40 degrees above the crate, 2.2 to 3.8 away, looking near its centre; on the orbit
            // once around at 22 degrees, 3 away.
            const float azimuth = orbit ? 6.2831853f * float(view) / float(views) : uniform(rng) * 6.2831853f;
            const float elevation = glm::radians(orbit ? 22.0f : 8.0f + 32.0f * uniform(rng));
            const float distance = orbit ? 3.0f : 2.2f + 1.6f * uniform(rng);
            const glm::vec3 look = glm::vec3(0.0f, 0.5f, 0.0f) +
                                   (orbit ? glm::vec3(0.0f) : glm::vec3(jitter(rng), jitter(rng), jitter(rng)));
            glm::vec3 eye = look + distance * glm::vec3(std::sin(azimuth) * std::cos(elevation), std::sin(elevation),
                                                        std::cos(azimuth) * std::cos(elevation));
            eye.y = std::max(eye.y, 0.15f);
            CameraMVP& ubo = renderer.getCamera()->ubo;
            ubo.model = glm::mat4(1.0f);
            ubo.view = glm::lookAt(eye, look, glm::vec3(0.0f, 1.0f, 0.0f));
            ubo.proj = glm::perspective(glm::radians(kFovDegrees), 1.0f, 0.05f, 100.0f);
            ubo.proj[1][1] *= -1.0f;
            ubo.cameraWorldPos = glm::vec4(eye, 1.0f);
            renderer.getCamera()->update(0);
            neural::GBufferFrame frame;
            frame.width = frame.height = size;
            frame.depthFar = kDepthFar;
            std::vector<uint8_t>* buffers[] = {&frame.colour,  &frame.classes, nullptr,
                                               &frame.depth16, &frame.u,       &frame.v};
            for (size_t k = 0; k < std::size(outputs); ++k) {
                const auto& [kind, output] = outputs[k];
                renderer.setOutput(output, 0.0f, kDepthFar);
                graph.submitAndWait(vc.getGraphicsQueue(), 0);
                const auto pixels = readImage(vc, target->getImage(0), extent);
                char name[64];
                std::snprintf(name, sizeof(name), "classical_%s%04u.png", kind, view);
                writeImageBgra(outDir / "frames" / name, pixels, size, size);
                if (buffers[k])
                    *buffers[k] = pixels;
            }
            if (enhancer) {
                // The sample as the training saw it: the inverse depth stretched up to the nearest point.
                const auto inputs =
                    neural::photoEnhancerInputs(frame, neural::PhotoEnhancerSample{sampleSize, sampleSize, 1.0f});
                const auto image = enhancer->run(inputs);
                const size_t pixels = size_t(sampleSize) * sampleSize;
                std::vector<uint8_t> rgb(3 * pixels);
                for (size_t p = 0; p < pixels; ++p)
                    for (size_t c = 0; c < 3; ++c)
                        rgb[3 * p + c] = uint8_t(std::lround(std::clamp(image[c * pixels + p], 0.0f, 1.0f) * 255.0f));
                char name[64];
                std::snprintf(name, sizeof(name), "enhanced_%04u.png", view);
                writeImage(outDir / "frames" / name, rgb.data(), sampleSize, sampleSize, 3);
            }
            cameras << view << "," << eye.x << "," << eye.y << "," << eye.z << "," << look.x << "," << look.y << ","
                    << look.z << ",0.05," << kFovDegrees << ",0," << kDepthFar << "\n";
        }
        std::cout << views << " views in " << outDir << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "crate_gbuffer_example: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
