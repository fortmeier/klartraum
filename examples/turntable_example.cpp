// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Renders a Gaussian splatting scene headlessly from a camera that orbits the
// scene's up axis exactly once, and writes one PPM image per frame, e.g. to
// assemble a looping animation (scripts/site/make_lantern_animation.sh).
//
//   turntable_example --file data/lantern.spz --flip-y --frames 72 --width 480 --height 360
//                     --camera-position 0.55 0.48 0.69 [--distance D]
//                     --out-dir build/TestingOutput/turntable
//
// The camera orbits the centre of the scene (the per-axis median of the
// Gaussian positions, so a few stray splats do not pull it off the object).
// --camera-position is the start position relative to that centre; --distance
// overrides its length and so how much of the image the scene fills.
//
// Frame i is seen from azimuth start + 2*pi*i/frames, so the frames cover one
// full turn without repeating the first view at the end. After the frames, the
// view after a full turn is rendered as well and compared with frame 0; the
// program fails if they differ, because the animation would then not loop
// seamlessly.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "klartraum/headless_frontend.hpp"
#include "klartraum/gaussian_splatting_factory.hpp"
#include "klartraum/gaussian_data_standard.hpp"
#include "klartraum/interface_camera_orbit.hpp"
#include "klartraum/offscreen_target.hpp"

namespace {

constexpr double kPi = 3.14159265358979323846;

// Reads an OffscreenTarget image back as tightly-packed BGRA bytes. The
// backends leave an OffscreenTarget in TRANSFER_SRC_OPTIMAL.
std::vector<uint8_t> readImage(klartraum::VulkanContext& vc, VkImage image, VkExtent2D extent) {
    const VkDeviceSize bytes = VkDeviceSize(extent.width) * extent.height * 4;
    VkBuffer buf;
    VkDeviceMemory mem;
    vc.createBuffer(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, buf, mem);
    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = vc.getCommandPool();
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(vc.getDevice(), &ai, &cmd);
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    VkBufferImageCopy region{};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = {extent.width, extent.height, 1};
    vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buf, 1, &region);
    vkEndCommandBuffer(cmd);
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vkQueueSubmit(vc.getGraphicsQueue(), 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(vc.getGraphicsQueue());
    vkFreeCommandBuffers(vc.getDevice(), vc.getCommandPool(), 1, &cmd);

    void* data;
    vkMapMemory(vc.getDevice(), mem, 0, bytes, 0, &data);
    std::vector<uint8_t> result(static_cast<const uint8_t*>(data), static_cast<const uint8_t*>(data) + bytes);
    vkUnmapMemory(vc.getDevice(), mem);
    vkFreeMemory(vc.getDevice(), mem, nullptr);
    vkDestroyBuffer(vc.getDevice(), buf, nullptr);
    return result;
}

void writePPM(const std::filesystem::path& path, const std::vector<uint8_t>& bgra, VkExtent2D extent) {
    std::ofstream out(path, std::ios::binary);
    out << "P6\n" << extent.width << " " << extent.height << "\n255\n";
    for (size_t i = 0; i < bgra.size(); i += 4) {
        const char rgb[3] = {char(bgra[i + 2]), char(bgra[i + 1]), char(bgra[i])};
        out.write(rgb, 3);
    }
}

// Per-axis median of the Gaussian centres.
glm::vec3 sceneCentre(const std::vector<klartraum::Gaussian3D>& gaussians) {
    glm::vec3 centre(0.0f);
    std::vector<float> values(gaussians.size());
    for (int axis = 0; axis < 3; ++axis) {
        for (size_t i = 0; i < gaussians.size(); ++i) {
            values[i] = gaussians[i].position[axis];
        }
        auto middle = values.begin() + values.size() / 2;
        std::nth_element(values.begin(), middle, values.end());
        centre[axis] = *middle;
    }
    return centre;
}

double meanAbsDifference(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
    double sum = 0.0;
    size_t count = 0;
    for (size_t i = 0; i < a.size(); i += 4) {
        for (size_t c = 0; c < 3; ++c) {
            sum += std::abs(int(a[i + c]) - int(b[i + c]));
            ++count;
        }
    }
    return count ? sum / double(count) : 0.0;
}

} // namespace

int main(int argc, char** argv) {
    std::string spzFile = "./data/lantern.spz";
    std::filesystem::path outDir = "build/TestingOutput/turntable";
    uint32_t frames = 72;
    uint32_t width = 480;
    uint32_t height = 360;
    double distanceOverride = 0.0;
    bool flipY = false;
    glm::vec3 cameraPosition(0.55f, 0.48f, 0.69f);
    klartraum::GsplatBackend backend = klartraum::GsplatBackend::Compute;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--file" && i + 1 < argc) {
            spzFile = argv[++i];
        } else if (arg == "--out-dir" && i + 1 < argc) {
            outDir = argv[++i];
        } else if (arg == "--frames" && i + 1 < argc) {
            frames = uint32_t(std::stoul(argv[++i]));
        } else if (arg == "--width" && i + 1 < argc) {
            width = uint32_t(std::stoul(argv[++i]));
        } else if (arg == "--height" && i + 1 < argc) {
            height = uint32_t(std::stoul(argv[++i]));
        } else if (arg == "--distance" && i + 1 < argc) {
            distanceOverride = std::stod(argv[++i]);
        } else if (arg == "--flip-y") {
            flipY = true;
        } else if (arg == "--camera-position" && i + 3 < argc) {
            cameraPosition.x = std::stof(argv[++i]);
            cameraPosition.y = std::stof(argv[++i]);
            cameraPosition.z = std::stof(argv[++i]);
        } else if (arg == "--backend" && i + 1 < argc) {
            std::string value = argv[++i];
            backend = value == "raster" ? klartraum::GsplatBackend::Raster : klartraum::GsplatBackend::Compute;
        } else {
            std::cerr << "unknown or incomplete argument: " << arg << std::endl;
            return 1;
        }
    }

    const double distance = distanceOverride > 0.0 ? distanceOverride : double(glm::length(cameraPosition));
    if (frames == 0 || width == 0 || height == 0 || !(glm::length(cameraPosition) > 0.0f)) {
        std::cerr << "--frames, --width and --height must be positive, --camera-position must differ from the origin"
                  << std::endl;
        return 1;
    }

    // Only the azimuth changes, so the camera circles the vertical (Y) axis
    // through the scene centre at a fixed height.
    const double startAzimuth = std::atan2(cameraPosition.z, cameraPosition.x);
    const double elevation = -std::asin(cameraPosition.y / glm::length(cameraPosition));

    const std::vector<klartraum::Gaussian3D> gaussians = klartraum::loadGaussiansSpz(spzFile, flipY);
    if (gaussians.empty()) {
        std::cerr << "no Gaussians loaded from " << spzFile << std::endl;
        return 1;
    }
    const glm::vec3 centre = sceneCentre(gaussians);
    std::cout << "scene centre: " << centre.x << " " << centre.y << " " << centre.z << std::endl;

    klartraum::HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();

    const VkExtent2D extent{width, height};
    auto target = std::make_shared<klartraum::OffscreenTarget>(vc, extent, 1u);
    auto cameraUBO = std::make_shared<klartraum::CameraUboType>();

    klartraum::InterfaceCameraOrbit orbit(klartraum::InterfaceCameraOrbit::UpDirection::Y);
    orbit.initialize(vc);
    // The orbit camera translates the world by its position before rotating
    // it about the origin, so -centre puts the scene centre at the pivot.
    orbit.setPosition(-centre);
    orbit.setElevation(elevation);
    orbit.setDistance(distance);
    orbit.setProjectionAspectRatio(float(width) / float(height));

    auto model = std::make_shared<klartraum::GaussianDataStandard>(vc, gaussians);
    auto splatting = klartraum::createGaussianSplatting(vc, backend, target, cameraUBO, model);

    std::filesystem::create_directories(outDir);

    std::vector<uint8_t> firstFrame;
    double loopDifference = 0.0;
    {
        klartraum::ComputeGraph graph(vc, 1);
        graph.compileFrom(splatting);

        auto renderAt = [&](double azimuth) {
            orbit.setAzimuth(azimuth);
            orbit.update(cameraUBO->ubo);
            cameraUBO->update(0);
            graph.submitAndWait(vc.getGraphicsQueue(), 0);
            return readImage(vc, target->getImage(0), extent);
        };

        for (uint32_t i = 0; i < frames; ++i) {
            auto pixels = renderAt(startAzimuth + 2.0 * kPi * double(i) / double(frames));
            char name[32];
            std::snprintf(name, sizeof(name), "frame_%04u.ppm", i);
            writePPM(outDir / name, pixels, extent);
            if (i == 0) {
                firstFrame = std::move(pixels);
            }
        }

        loopDifference = meanAbsDifference(renderAt(startAzimuth + 2.0 * kPi), firstFrame);
    }
    splatting.reset();
    target.reset();

    std::cout << "wrote " << frames << " frames of " << width << "x" << height << " to " << outDir.string()
              << std::endl;
    std::cout << "mean difference between frame 0 and the view after a full turn: " << loopDifference << std::endl;

    // The full-turn view differs from frame 0 only by floating-point rounding
    // of the azimuth, so anything above a small fraction of a grey level means
    // the orbit does not close.
    if (loopDifference > 0.5) {
        std::cerr << "the orbit does not return to its start; the animation would not loop" << std::endl;
        return 1;
    }
    return 0;
}
