// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// The bonsai landscape's forest fly-in rendered photo-realistically in Klartraum: every frame is rasterized at 1280 x
// 720 into G-buffers (colour, classes, 16-bit depth, surface coordinates u, v and object id) and downsampled to the
// 832 x 464 samples of training, a learnable surface texture (neural::hashTexture(), a multi-resolution hash grid over
// (u, v, id)) gives every pixel its features, and the photo enhancer's generator (an ONNX graph) turns G-buffers and
// features into the frame -- the network scripts/neural/train_photo_enhancer.py --texture trains from Qwen-Image
// photographs.
//
//   photo_enhancer_example --model build/TestingOutput/gan_runs/v6/photo_enhancer --out-dir build/TestingOutput/pe
//
// Frames go to <out-dir>/frames/frame_NNNN.png (enhanced) and render_NNNN.png (the downsampled rendering).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <glm/gtc/matrix_transform.hpp>

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/tensorelement.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/image_io.hpp"
#include "klartraum/neural/photo_enhancer.hpp"
#include "klartraum/offscreen_target.hpp"
#include "klartraum/onnx/onnx_network.hpp"
#include "klartraum/tree/bonsai_flight.hpp"
#include "klartraum/tree/bonsai_renderer.hpp"

using namespace klartraum;

namespace {

constexpr uint32_t kRenderWidth = 1280, kRenderHeight = 720; // the training flights' frames
constexpr uint32_t kWidth = 832, kHeight = 464;              // the training's sample size

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

void setCamera(CameraMVP& ubo, glm::vec3 eye, glm::vec3 target, float aspect, float nearPlane) {
    ubo.model = glm::mat4(1.0f);
    ubo.view = glm::lookAt(eye, target, glm::vec3(0.0f, 1.0f, 0.0f));
    ubo.proj = glm::perspective(glm::radians(40.0f), aspect, nearPlane, 50.0f);
    ubo.proj[1][1] *= -1.0f;
    ubo.cameraWorldPos = glm::vec4(eye, 1.0f);
}

} // namespace

int main(int argc, char** argv) {
    std::filesystem::path model = "build/TestingOutput/gan_runs/v6/photo_enhancer";
    std::filesystem::path outDir = "build/TestingOutput/photo_enhancer";
    BonsaiSceneConfig sceneConfig;
    sceneConfig.scale.s = 1.0f;
    float azimuth = -2.09f, seconds = 24.0f, fps = 30.0f, season = 2.0f;
    uint32_t every = 1, first = 0, last = UINT32_MAX;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const bool hasValue = i + 1 < argc;
        if (arg == "--model" && hasValue)
            model = argv[++i];
        else if (arg == "--out-dir" && hasValue)
            outDir = argv[++i];
        else if (arg == "--seed" && hasValue)
            sceneConfig.seed = uint32_t(std::stoul(argv[++i]));
        else if (arg == "--azimuth" && hasValue)
            azimuth = std::stof(argv[++i]);
        else if (arg == "--season" && hasValue)
            season = std::stof(argv[++i]);
        else if (arg == "--fly-in" && hasValue)
            seconds = std::stof(argv[++i]);
        else if (arg == "--every" && hasValue)
            every = std::max(1, std::stoi(argv[++i]));
        else if (arg == "--from" && hasValue)
            first = uint32_t(std::stoul(argv[++i]));
        else if (arg == "--to" && hasValue)
            last = uint32_t(std::stoul(argv[++i]));
        else {
            std::cerr << "unknown argument " << arg << std::endl;
            return 1;
        }
    }
    try {
        HeadlessFrontend frontend;
        VulkanContext& vc = frontend.getKlartraumEngine().getVulkanContext();
        const BonsaiScene scene = makeBonsaiScene(sceneConfig);
        const glm::vec3 start = scene.focus + 4.2f * glm::vec3(std::sin(azimuth) * std::cos(0.32f), std::sin(0.32f),
                                                               std::cos(azimuth) * std::cos(0.32f));
        ForestApproachConfig flightConfig;
        flightConfig.seconds = seconds;
        flightConfig.season = season;
        ForestFlight flight(scene, start, scene.focus, findForestApproach(scene, flightConfig), flightConfig);
        if (!flight.isValid())
            throw std::runtime_error("no place found to fly into a forest");

        const VkExtent2D extent{kRenderWidth, kRenderHeight};
        auto target = std::make_shared<OffscreenTarget>(vc, extent, 1);
        BonsaiRenderConfig config;
        config.extent = extent;
        BonsaiRenderer bonsai(vc, scene, target, config);
        ComputeGraph renderGraph(vc, 1);
        renderGraph.compileFrom(bonsai.getOutput());
        bonsai.initialize(1);
        bonsai.setSeason(season);

        neural::PhotoEnhancer enhancer(vc, model.string(), kWidth, kHeight);
        if (enhancer.hasSeason())
            enhancer.setSeason(season);
        const size_t pixels = size_t(kWidth) * kHeight;

        std::filesystem::create_directories(outDir / "frames");
        const uint32_t frames = uint32_t(std::lround(seconds * fps));
        float depthFar = 50.0f; // the depth window, fitted per frame as tree_bonsai_example --gbuffers does
        const auto render = [&](BonsaiOutput output, float near = 0.0f, float far = 1.0f) {
            bonsai.setOutput(output, near, far);
            renderGraph.submitAndWait(vc.getGraphicsQueue(), 0);
            return readImage(vc, target->getImage(0), extent);
        };
        for (uint32_t frame = 0; frame < std::min(frames, last); ++frame) {
            const FlyInPose pose = flight.at(float(frame) / float(std::max(frames - 1, 1u)));
            setCamera(bonsai.getCamera()->ubo, pose.eye, pose.lookAt, float(kRenderWidth) / float(kRenderHeight),
                      pose.nearPlane);
            bonsai.getCamera()->update(0);
            bonsai.setView(pose.eye);
            bonsai.setTime(float(frame) / fps, 0.5f, season,
                           glm::vec4(glm::normalize(glm::vec3(1.0f, 0.0f, 0.35f)), 0.5f));
            // Tiles of the placement come a few per frame; a second frame lets them keep up with the flight.
            renderGraph.submitAndWait(vc.getGraphicsQueue(), 0);
            bonsai.setView(pose.eye);
            if (frame < first || (frame - first) % every != 0) {
                renderGraph.submitAndWait(vc.getGraphicsQueue(), 0);
                continue;
            }
            const auto started = std::chrono::steady_clock::now();
            const auto colour = render(BonsaiOutput::Colour);
            neural::GBufferFrame gbuffers;
            gbuffers.width = kRenderWidth;
            gbuffers.height = kRenderHeight;
            gbuffers.colour = colour;
            gbuffers.classes = render(BonsaiOutput::Classes);
            // The depth window as tree_bonsai_example --gbuffers fits it: the 98th percentile of the covered
            // pixels' distances near 80 % of the grey range, starting from the previous frame's window.
            for (int attempt = 0; attempt < 12; ++attempt) {
                const auto depth = render(BonsaiOutput::Depth, 0.0f, depthFar);
                std::vector<uint8_t> covered;
                for (size_t p = 0; p < depth.size(); p += 4)
                    if (gbuffers.classes[p] | gbuffers.classes[p + 1] | gbuffers.classes[p + 2])
                        covered.push_back(depth[p]);
                if (covered.empty())
                    break;
                const size_t rank = covered.size() * 98 / 100;
                std::nth_element(covered.begin(), covered.begin() + rank, covered.end());
                const float percentile = float(covered[rank]);
                float next = percentile >= 255.0f  ? depthFar * 2.0f
                             : percentile < 150.0f ? depthFar * std::max(percentile, 4.0f) / 205.0f
                                                   : depthFar;
                next = std::min(next, 50.0f);
                if (next == depthFar)
                    break;
                depthFar = next;
            }
            gbuffers.depth16 = render(BonsaiOutput::Depth16, 0.0f, depthFar);
            gbuffers.depthFar = depthFar;
            gbuffers.u = render(BonsaiOutput::CoordinatesU);
            gbuffers.v = render(BonsaiOutput::CoordinatesV);
            const auto inputs = neural::photoEnhancerInputs(gbuffers, neural::PhotoEnhancerSample{kWidth, kHeight});
            const auto image = enhancer.run(inputs);
            const auto& values = inputs.channels;

            std::vector<uint8_t> rgb(3 * pixels), rendering(3 * pixels);
            for (size_t p = 0; p < pixels; ++p)
                for (int c = 0; c < 3; ++c) {
                    rgb[3 * p + c] = uint8_t(std::lround(std::clamp(image[c * pixels + p], 0.0f, 1.0f) * 255.0f));
                    rendering[3 * p + c] = uint8_t(std::lround(values[c * pixels + p] * 255.0f));
                }
            char name[64];
            std::snprintf(name, sizeof(name), "frame_%04u.png", frame);
            writeImage(outDir / "frames" / name, rgb.data(), kWidth, kHeight, 3);
            std::snprintf(name, sizeof(name), "render_%04u.png", frame);
            writeImage(outDir / "frames" / name, rendering.data(), kWidth, kHeight, 3);
            std::cout << "frame " << frame << " in "
                      << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() << " s"
                      << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "photo_enhancer_example: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
