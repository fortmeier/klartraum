// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Renders a 3D skeleton headlessly as ControlNet OpenPose pose maps, from a
// camera that circles it once, and writes one PPM image per frame.
//
//   openpose_skeleton_example --frames 36 --width 512 --height 512 --distance 3.2
//                             --out-dir build/TestingOutput/openpose_skeleton
//
// The skeleton is a fixed standing pose in metres (y up, facing +z). The
// joints live in a storage buffer that OpenPoseSkeletonRenderer projects with
// the orbit camera, so the drawing follows the 3D pose from every side. The
// first frame looks at the front, where the right limbs (warm colours) appear
// on the left of the image; seen from behind, they appear on the right.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "klartraum/computegraph/bufferelement.hpp"
#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/renderpass.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/interface_camera_orbit.hpp"
#include "klartraum/offscreen_target.hpp"
#include "klartraum/openpose_skeleton_renderer.hpp"
#include "klartraum/vulkan_buffer.hpp"

namespace {

constexpr double kPi = 3.14159265358979323846;

// A standing pose with the arms slightly raised, in COCO-18 order.
const std::array<glm::vec3, klartraum::openpose::kJointCount> kStandingPose = {{
    {0.00f, 1.62f, 0.08f},   // nose
    {0.00f, 1.45f, 0.00f},   // neck
    {-0.19f, 1.43f, 0.00f},  // right shoulder
    {-0.30f, 1.17f, 0.04f},  // right elbow
    {-0.38f, 0.93f, 0.16f},  // right wrist
    {0.19f, 1.43f, 0.00f},   // left shoulder
    {0.34f, 1.19f, -0.02f},  // left elbow
    {0.47f, 0.98f, 0.06f},   // left wrist
    {-0.10f, 0.95f, 0.00f},  // right hip
    {-0.12f, 0.52f, 0.06f},  // right knee
    {-0.13f, 0.08f, -0.02f}, // right ankle
    {0.10f, 0.95f, 0.00f},   // left hip
    {0.14f, 0.53f, -0.04f},  // left knee
    {0.16f, 0.08f, -0.10f},  // left ankle
    {-0.035f, 1.66f, 0.06f}, // right eye
    {0.035f, 1.66f, 0.06f},  // left eye
    {-0.075f, 1.64f, 0.00f}, // right ear
    {0.075f, 1.64f, 0.00f},  // left ear
}};

std::vector<uint8_t> readImage(klartraum::VulkanContext& vc, VkImage image, VkImageLayout layout, VkExtent2D extent) {
    const VkDeviceSize bytes = VkDeviceSize(extent.width) * extent.height * 4;
    VkBuffer buffer;
    VkDeviceMemory memory;
    vc.createBuffer(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, buffer, memory);
    vc.submitImmediate([&](VkCommandBuffer commandBuffer) {
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {extent.width, extent.height, 1};
        vkCmdCopyImageToBuffer(commandBuffer, image, layout, buffer, 1, &region);
    });
    void* data;
    vkMapMemory(vc.getDevice(), memory, 0, bytes, 0, &data);
    std::vector<uint8_t> result(static_cast<const uint8_t*>(data), static_cast<const uint8_t*>(data) + bytes);
    vkUnmapMemory(vc.getDevice(), memory);
    vkDestroyBuffer(vc.getDevice(), buffer, nullptr);
    vkFreeMemory(vc.getDevice(), memory, nullptr);
    return result;
}

void writePpm(const std::filesystem::path& path, const std::vector<uint8_t>& bgra, VkExtent2D extent) {
    std::ofstream out(path, std::ios::binary);
    out << "P6\n" << extent.width << " " << extent.height << "\n255\n";
    for (size_t i = 0; i < bgra.size(); i += 4) {
        const char rgb[3] = {char(bgra[i + 2]), char(bgra[i + 1]), char(bgra[i])};
        out.write(rgb, 3);
    }
}

} // namespace

int main(int argc, char** argv) {
    std::filesystem::path outDir = "build/TestingOutput/openpose_skeleton";
    uint32_t frames = 36;
    uint32_t width = 512;
    uint32_t height = 512;
    float distance = 3.2f;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--out-dir" && i + 1 < argc) {
            outDir = argv[++i];
        } else if (arg == "--frames" && i + 1 < argc) {
            frames = uint32_t(std::stoul(argv[++i]));
        } else if (arg == "--width" && i + 1 < argc) {
            width = uint32_t(std::stoul(argv[++i]));
        } else if (arg == "--height" && i + 1 < argc) {
            height = uint32_t(std::stoul(argv[++i]));
        } else if (arg == "--distance" && i + 1 < argc) {
            distance = std::stof(argv[++i]);
        } else {
            std::cerr << "unknown or incomplete argument: " << arg << std::endl;
            return 1;
        }
    }
    if (frames == 0 || width == 0 || height == 0 || !(distance > 0.0f)) {
        std::cerr << "--frames, --width, --height and --distance must be positive" << std::endl;
        return 1;
    }

    klartraum::HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();

    const VkExtent2D extent{width, height};
    auto target = std::make_shared<klartraum::OffscreenTarget>(vc, extent, 1u);
    auto cameraUBO = std::make_shared<klartraum::CameraUboType>();
    auto joints = std::make_shared<klartraum::BufferElement<klartraum::VulkanBuffer<glm::vec4>>>(
        vc, klartraum::openpose::kJointCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto renderer = std::make_shared<klartraum::OpenPoseSkeletonRenderer>(joints, 1, extent);

    auto renderPass = std::make_shared<klartraum::RenderPass>(vc.getSwapChainImageFormat(), extent);
    renderPass->setInput(target, 0);
    renderPass->setInput(cameraUBO, 1);
    renderPass->addDrawComponent(renderer);
    renderPass->addComputeDependency(joints);

    klartraum::InterfaceCameraOrbit orbit(klartraum::InterfaceCameraOrbit::UpDirection::Y);
    orbit.initialize(vc);
    // The orbit camera pivots about minus its position: look at the hips.
    orbit.setPosition({0.0f, -0.9f, 0.0f});
    orbit.setElevation(-0.15f);
    orbit.setDistance(distance);
    orbit.setProjectionAspectRatio(float(width) / float(height));

    std::filesystem::create_directories(outDir);
    {
        klartraum::ComputeGraph graph(vc, 1);
        graph.compileFrom(renderPass);

        std::vector<glm::vec4> pose;
        for (const auto& joint : kStandingPose) {
            pose.emplace_back(joint, 1.0f);
        }
        joints->getBuffer(0).memcopyFrom(pose);

        for (uint32_t frame = 0; frame < frames; ++frame) {
            orbit.setAzimuth(kPi / 2.0 + 2.0 * kPi * double(frame) / double(frames));
            orbit.update(cameraUBO->ubo);
            cameraUBO->update(0);
            graph.submitAndWait(vc.getGraphicsQueue(), 0);
            char name[32];
            std::snprintf(name, sizeof(name), "frame_%04u.ppm", frame);
            writePpm(outDir / name, readImage(vc, target->getImage(0), *target->getFinalLayoutOverride(), extent),
                     extent);
        }
    }

    std::cout << "wrote " << frames << " pose maps of " << width << "x" << height << " to " << outDir.string()
              << std::endl;
    return 0;
}
