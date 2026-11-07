// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Renders "klartraum tech demo 001", a promo video of the differentiable rendering and the procedural trees,
// headlessly. Chapters, one continuous camera:
//   0. Title and morph: on white, the black "klartraum" logo (Gaussians in Unbounded) and a large black sphere above
//      it, which at once morphs into the pumpkin by differentiable mesh optimization on the masked photographs of
//      the pumpkin (raster::MeshMorph): every frame runs optimization steps whose gradients flow from the
//      photographs into the mesh's shape and texture, in the frame's compute graph. Beside it, the photograph in use
//      and the training's rendering of the mesh from its camera. The jack-o'-lantern faces the camera.
//   1. Drop: the pumpkin falls onto the splatted bonsai pot with its soil and the lantern, shrinking to its size on
//      the soil and turning to face the camera; the camera follows.
//   2. Growth: the bonsai pine grows from a seedling, its trunk extending and branching out step by step.
//   3. Style transfer: the summer afternoon holds while a painted style fades in (with --style or --field; with
//      --panels a training sample of the style beside it).
//   4. Days: the afternoon turns into night, then four days, one season each, stormy in fall and winter (with
//      --panels the style's targets of the season by day and night beside them); the background is white by day and
//      black by night.
// Each chapter shows a caption with the work it builds on, lower left.
//
//   techdemo001_example [--pot FILE] [--photos DIR] [--width 1280] [--height 720] [--fps 30]
//                       [--from 0] [--to 4] [--step 1] [--out-dir build/TestingOutput/techdemo001]
//                       [--style MODEL.onnx | --field STEM] [--panels DIR]
//
// --pot is the object.spz of splat_training_example for the bonsai pot (default:
// build/TestingOutput/splat_training/bonsai/object.spz), --photos the pumpkin's photographs as prepared by
// scripts/mesh_fitting/prepare_data.py (default: build/TestingOutput/real_data/pumpkin_closed_1_512). Frames go to
// <out-dir>/frames/frame_XXXX.ppm, numbered through the whole video; --from/--to select chapters, --step renders
// every n-th frame for previews.

#include <algorithm>
#include <functional>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <random>
#include <tuple>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../3rdparty/glfw/deps/stb_image_write.h"

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/demo_sky_marks.hpp"
#include "klartraum/demo_timeline.hpp"
#include "klartraum/field_style.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/offscreen_target.hpp"
#include "klartraum/raster/differentiable_shading.hpp"
#include "klartraum/raster/fitting_scene.hpp"
#include "klartraum/raster/mesh.hpp"
#include "klartraum/raster/mesh_morph.hpp"
#include "klartraum/rendering/gaussian_data_standard.hpp"
#include "klartraum/rendering/gaussian_splatting_factory.hpp"
#include "klartraum/rendering/text_gaussians.hpp"
#include "klartraum/style_transfer.hpp"
#include "klartraum/tree/bonsai_renderer.hpp"
#include "klartraum/tree/bonsai_scene.hpp"
#include "klartraum/tree/tree_sky.hpp"

namespace {

using namespace klartraum;

constexpr float kFov = 40.0f; // vertical field of view of every camera, degrees

struct Chapter {
    float seconds;
    std::vector<std::string> caption; // first line large, the others small
};

// Whether the days show a black sun by day and a white crescent moon by night where the light comes from.
constexpr bool kSkyMarks = false;

const std::vector<Chapter> kChapters = {
    {5.0f, {"Differentiable rasterization", "Laine et al. 2020, automatic differentiation in Slang"}},
    {2.5f, {"Gaussian splatting", "Kerbl et al. 2023, trained on 38 photographs"}},
    {5.0f, {"GPU tree generation", "Kuth et al. 2025, after Weber and Penn 1995"}},
    {4.0f, {"Watercolor style transfer", "per pixel from G-buffers and world-anchored features, 185 painted pairs"}},
    {18.0f, {"Days and seasons", "one compute graph submission per frame"}},
};

// The frame's pixels as BGRA bytes, from an image in `layout`.
std::vector<uint8_t> readImage(VulkanContext& vc, VkImage image, VkImageLayout layout, VkExtent2D extent) {
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

// A camera pose: where it stands and what it looks at.
struct Pose {
    glm::vec3 eye, target;
};

Pose mix(const Pose& a, const Pose& b, float t) { return {glm::mix(a.eye, b.eye, t), glm::mix(a.target, b.target, t)}; }

CameraMVP cameraOf(const Pose& pose, float aspect) {
    CameraMVP ubo{};
    ubo.model = glm::mat4(1.0f);
    ubo.view = glm::lookAt(pose.eye, pose.target, glm::vec3(0.0f, 1.0f, 0.0f));
    ubo.proj = glm::perspective(glm::radians(kFov), aspect, 0.05f, 60.0f);
    ubo.proj[1][1] *= -1.0f;
    ubo.cameraWorldPos = glm::vec4(pose.eye, 1.0f);
    return ubo;
}

float smooth(float t) {
    t = std::clamp(t, 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// The background by day and night: white while the sun is up, black in the night, smooth in the twilight.
float backgroundAt(float timeOfDay, float season) {
    const float elevation = treeSkyAt(timeOfDay, season).sunElevation;
    return smooth((elevation + 0.2f) / 0.6f);
}

// A caption: its lines rasterized, drawn lower left in the colour opposite the background.
struct Caption {
    std::vector<CoverageMask> lines;
};

Caption makeCaption(const Chapter& chapter, const std::shared_ptr<Font>& title, const std::shared_ptr<Font>& mono,
                    uint32_t height) {
    Caption caption;
    for (size_t i = 0; i < chapter.caption.size(); ++i) {
        TitleLayout layout;
        layout.align = TextAlign::Left;
        layout.pixelsPerUnit = float(height);
        layout.lines.push_back(
            {chapter.caption[i], i == 0 ? title : mono, i == 0 ? 500.0f : 400.0f, i == 0 ? 0.034f : 0.021f, 0.0f});
        caption.lines.push_back(rasterizeTitle(layout));
    }
    return caption;
}

void drawCaption(std::vector<uint8_t>& frame, VkExtent2D extent, const Caption& caption, float opacity, float ink) {
    if (opacity <= 0.0f)
        return;
    int y = int(extent.height) - int(0.05f * float(extent.height));
    for (auto line = caption.lines.rbegin(); line != caption.lines.rend(); ++line) {
        y -= int(line->height);
        blendCaption(frame, extent.width, extent.height, *line, int(0.04f * float(extent.width)), y, glm::vec3(ink),
                     opacity);
        y -= int(0.01f * float(extent.height));
    }
}

// Copies `source` (in TRANSFER_SRC_OPTIMAL) into `target`, which is left in TRANSFER_SRC_OPTIMAL as a renderer leaves
// its frame.
void copyImage(VulkanContext& vc, VkImage source, VkImage target, VkExtent2D extent) {
    vc.submitImmediate([&](VkCommandBuffer commandBuffer) {
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = target;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                             nullptr, 0, nullptr, 1, &barrier);
        VkImageCopy region{};
        region.srcSubresource = region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.extent = {extent.width, extent.height, 1};
        vkCmdCopyImage(commandBuffer, source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, target,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0,
                             nullptr, 0, nullptr, 1, &barrier);
    });
}

// Writes `count` renderings of the bonsai scene (the trees fully grown, no wind) from random cameras at random seasons
// and times of day to `directory` as PNG files, each with its G-buffers (class_XXXX.png: the surfaces' classes as
// tree::treeClassColor(); depth_XXXX.png: their distance from the camera as grey from kDepthNear to kDepthFar;
// coverage_XXXX.png: how much of every pixel the surfaces cover, as grey; field_depth_XXXX.png: their distance over
// black as FieldStyle reads it, in the window of kFieldBoxLow..kFieldBoxHigh, with the finer steps of the three
// channels, tree::treeGBufferColor()),
// and dataset.yaml giving each one's camera, light, season and time. The background is white by day and black by night
// as in the video; there are no captions and no sun or moon. With `sequences`, also that many short sequences of
// kSequenceFrames frames each (a slow orbit step and a few minutes of the day from frame to frame) to
// sequences/seq_XXX/, with sequences.yaml, e.g. for training and measuring temporal stability.
constexpr float kDepthNear = 2.0f, kDepthFar = 16.0f;
// The box FieldStyle's world grids cover: the pot, the soil and the fully grown tree.
constexpr glm::vec3 kFieldBoxLow(-1.7f, -0.55f, -1.3f), kFieldBoxHigh(1.7f, 3.7f, 1.2f);
constexpr uint32_t kSequenceFrames = 8;

// The side panels of the days (--panels, scripts/neural/prepare_style_panels.py): during the style's demonstration a
// training sample (its rendering, classes, depth and watercolor target) in a 2 x 2 grid, then the watercolor targets
// of the season shown by day and by night. Labels are black on light backgrounds and white on dark ones.
class StylePanels {
public:
    StylePanels(const std::filesystem::path& folder, VkExtent2D extent, const std::shared_ptr<Font>& mono)
        : extent(extent) {
        for (const char* name : {"rendering", "classes", "depth", "target"})
            sample.push_back(read(folder / ("sample_" + std::string(name) + ".ppm")));
        for (const char* season : {"winter", "spring", "summer", "fall"})
            for (const char* kind : {"day", "night"})
                targets.push_back(read(folder / ("target_" + std::string(season) + "_" + kind + ".ppm")));
        const auto label = [&](const std::string& text) {
            TitleLayout layout;
            layout.align = TextAlign::Left;
            layout.pixelsPerUnit = float(extent.height);
            layout.lines.push_back({text, mono, 400.0f, 0.021f, 0.0f});
            return rasterizeTitle(layout);
        };
        for (const char* name : {"rendering", "classes", "depth", "watercolor target"})
            sampleLabels.push_back(label(name));
        for (const char* season : {"winter", "spring", "summer", "fall"})
            for (const char* kind : {"day", "night"})
                targetLabels.push_back(label(std::string(season) + " " + kind + ", watercolor target"));
    }

    // Draws the panels `seconds` into the days, the demonstration lasting `demonstration` seconds.
    void draw(std::vector<uint8_t>& bgra, float seconds, float demonstration, float season, float background) const {
        const glm::vec3 ink(background < 0.5f ? 1.0f : 0.0f);
        const int margin = int(0.05f * float(extent.width)), gap = int(0.03f * float(extent.height));
        if (seconds < demonstration) {
            const float opacity = smooth((seconds - 0.2f) / 0.5f) * smooth((demonstration - seconds) / 0.4f);
            const Picture& first = sample[0];
            const int width = int(first.width), height = int(first.height), label = int(sampleLabels[0].height);
            const int left = int(extent.width) - margin - 2 * width - gap;
            const int top = (int(extent.height) - 2 * (label + gap / 2 + height) - gap) / 2;
            for (int k = 0; k < 4; ++k) {
                const int x = left + (k % 2) * (width + gap), y = top + (k / 2) * (label + gap / 2 + height + gap);
                blendCaption(bgra, extent.width, extent.height, sampleLabels[k], x, y, ink, opacity);
                blendPicture(bgra, extent.width, extent.height, sample[k].values, sample[k].width, sample[k].height, x,
                             y + label + gap / 2, opacity);
            }
            return;
        }
        const float opacity = smooth((seconds - demonstration - 0.2f) / 0.5f);
        // The season nearest the one shown (winter 0 = 4, spring 1.25, summer 2, fall 3.25).
        const float points[] = {0.0f, 1.25f, 2.0f, 3.25f};
        int nearest = 0;
        float best = 1e9f;
        for (int k = 0; k < 4; ++k) {
            const float d = std::fmod(std::abs(season - points[k]), 4.0f);
            if (std::min(d, 4.0f - d) < best) {
                best = std::min(d, 4.0f - d);
                nearest = k;
            }
        }
        const Picture& first = targets[0];
        const int width = int(first.width), height = int(first.height), label = int(targetLabels[0].height);
        const int x = int(extent.width) - margin - width;
        int y = (int(extent.height) - 2 * (label + gap / 2 + height) - gap) / 2;
        for (int kind = 0; kind < 2; ++kind) {
            const int k = 2 * nearest + kind;
            blendCaption(bgra, extent.width, extent.height, targetLabels[k], x, y, ink, opacity);
            blendPicture(bgra, extent.width, extent.height, targets[k].values, targets[k].width, targets[k].height, x,
                         y + label + gap / 2, opacity);
            y += label + gap / 2 + height + gap;
        }
    }

private:
    struct Picture {
        std::vector<float> values; // four floats per pixel, rgb and 1
        uint32_t width = 0, height = 0;
    };

    static Picture read(const std::filesystem::path& path) {
        Picture picture;
        picture.values = raster::readTexturePpm(path, picture.width, picture.height);
        return picture;
    }

    // The frame's size the panels are laid out in.
    VkExtent2D extent;
    // The training sample (rendering, classes, depth, target) and the targets (per season: day, night).
    std::vector<Picture> sample, targets;
    std::vector<CoverageMask> sampleLabels, targetLabels;
};

// Floats to and from a raw little-endian file (an empty vector if the file is missing).
void writeFloats(const std::filesystem::path& path, const std::vector<float>& values) {
    std::ofstream(path, std::ios::binary)
        .write(reinterpret_cast<const char*>(values.data()), std::streamsize(values.size() * sizeof(float)));
}

std::vector<float> readFloats(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
        return {};
    std::vector<float> values(size_t(file.tellg()) / sizeof(float));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(values.data()), std::streamsize(values.size() * sizeof(float)));
    return values;
}

// The conditions FieldStyle's network was trained with (train_watercolor_field.py: conditions()): time of day and
// season on circles, the directional light (the moon's counted less), the exposure and the background.
std::array<float, 7> fieldConditions(float timeOfDay, float season, float lightIntensity, bool sun, float exposure,
                                     float background) {
    const float tau = 6.2831853f, s = season / 4.0f;
    return {std::sin(tau * timeOfDay),
            std::cos(tau * timeOfDay),
            std::sin(tau * s),
            std::cos(tau * s),
            lightIntensity * (sun ? 1.0f : 0.3f),
            exposure / 3.0f,
            background};
}

void writeDataset(VulkanContext& vc, const BonsaiScene& scene, BonsaiRenderConfig config, VkExtent2D extent,
                  uint32_t count, uint32_t sequences, uint32_t seed, const std::filesystem::path& directory) {
    std::filesystem::create_directories(directory);
    auto target = std::make_shared<OffscreenTarget>(vc, extent, 1);
    config.extent = extent;
    BonsaiRenderer bonsai(vc, scene, target, config);
    ComputeGraph graph(vc, 1);
    graph.compileFrom(bonsai.getOutput());
    bonsai.initialize(1);
    const float aspect = float(extent.width) / float(extent.height);
    std::mt19937 random(seed);
    const auto uniform = [&](float lo, float hi) { return std::uniform_real_distribution<float>(lo, hi)(random); };
    const auto vec = [](glm::vec3 v) {
        std::ostringstream text;
        text << "[" << v.x << ", " << v.y << ", " << v.z << "]";
        return text.str();
    };
    const auto matrix = [](const glm::mat4& m) {
        std::ostringstream text;
        text << "[";
        for (int row = 0; row < 4; ++row)
            for (int column = 0; column < 4; ++column)
                text << m[column][row] << (row == 3 && column == 3 ? "]" : ", ");
        return text.str();
    };
    const auto seasonName = [](float season) {
        // The seasons the days show (BonsaiDays): spring 1.25, summer 2, fall 3.25, winter 4 (= 0).
        const std::pair<const char*, float> named[] = {
            {"winter", 0.0f}, {"spring", 1.25f}, {"summer", 2.0f}, {"fall", 3.25f}, {"winter", 4.0f}};
        const char* nearest = "winter";
        float best = 1e9f;
        for (const auto& [name, at] : named)
            if (std::abs(season - at) < best) {
                best = std::abs(season - at);
                nearest = name;
            }
        return nearest;
    };
    const auto readTarget = [&]() {
        return readImage(vc, target->getImage(0), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, extent);
    };
    const auto writePng = [&](const std::filesystem::path& path) {
        const std::vector<uint8_t> bgra = readTarget();
        std::vector<uint8_t> rgb(3 * size_t(extent.width) * extent.height);
        for (size_t p = 0; p < size_t(extent.width) * extent.height; ++p)
            for (int c = 0; c < 3; ++c)
                rgb[3 * p + c] = bgra[4 * p + 2 - c];
        stbi_write_png(path.string().c_str(), int(extent.width), int(extent.height), 3, rgb.data(),
                       int(3 * extent.width));
    };
    // Renders a view as colour, classes and depth into `folder` (image/class/depth + `suffix`.png) and writes its
    // YAML entry with `indent` spaces.
    const auto shoot = [&](std::ostream& yaml, const std::string& indent, const std::filesystem::path& folder,
                           const std::string& suffix, glm::vec3 eye, glm::vec3 lookAt, float timeOfDay, float season,
                           float seconds) {
        const Pose pose{eye, lookAt};
        bonsai.getCamera()->ubo = cameraOf(pose, aspect);
        bonsai.getCamera()->update(0);
        bonsai.setView(eye);
        bonsai.setTime(seconds, timeOfDay, season, glm::vec4(1.0f, 0.0f, 0.0f, 0.0f));
        const float background = backgroundAt(timeOfDay, season);
        bonsai.setOutput(BonsaiOutput::Colour);
        bonsai.setBackground(glm::vec3(background));
        graph.submitAndWait(vc.getGraphicsQueue(), 0);
        writePng(folder / ("image_" + suffix + ".png"));
        bonsai.setOutput(BonsaiOutput::Classes);
        bonsai.setBackground(glm::vec3(0.0f));
        graph.submitAndWait(vc.getGraphicsQueue(), 0);
        writePng(folder / ("class_" + suffix + ".png"));
        // How much of every pixel the surfaces cover: a depth rendering whose window lies in front of everything
        // shows every surface white, over black.
        bonsai.setOutput(BonsaiOutput::Depth, -2.0f, -1.0f);
        graph.submitAndWait(vc.getGraphicsQueue(), 0);
        writePng(folder / ("coverage_" + suffix + ".png"));
        bonsai.setOutput(BonsaiOutput::Depth, kDepthNear, kDepthFar);
        bonsai.setBackground(glm::vec3(1.0f));
        graph.submitAndWait(vc.getGraphicsQueue(), 0);
        writePng(folder / ("depth_" + suffix + ".png"));
        // The depth as FieldStyle reads it: over black, in the window of the field's box.
        const auto [fieldNear, fieldFar] = fieldDepthWindow(eye, kFieldBoxLow, kFieldBoxHigh);
        bonsai.setOutput(BonsaiOutput::Depth, fieldNear, fieldFar);
        bonsai.setBackground(glm::vec3(0.0f));
        graph.submitAndWait(vc.getGraphicsQueue(), 0);
        writePng(folder / ("field_depth_" + suffix + ".png"));
        bonsai.setOutput(BonsaiOutput::Colour);

        const TreeRenderSettings s = bonsai.getSettings().values().at(0);
        const TreeSky sky = treeSkyAt(timeOfDay, season);
        const bool sun = glm::dot(sky.lightDirection, sky.sunDirection) > 0.999f;
        const CameraMVP& camera = bonsai.getCamera()->ubo;
        const int minutes = int(std::lround(timeOfDay * 24.0f * 60.0f)) % (24 * 60);
        std::ostringstream clock;
        clock << std::setw(2) << std::setfill('0') << minutes / 60 << ":" << std::setw(2) << std::setfill('0')
              << minutes % 60;
        const std::string& i = indent;
        yaml << i << "- file: image_" << suffix << ".png\n"
             << i << "  classes: class_" << suffix << ".png\n"
             << i << "  depth: depth_" << suffix << ".png\n"
             << i << "  coverage: coverage_" << suffix << ".png\n"
             << i << "  field_depth: field_depth_" << suffix << ".png\n"
             << i << "  field_depth_near: " << fieldNear << "\n"
             << i << "  field_depth_far: " << fieldFar << "\n"
             << i << "  season: " << season << "\n"
             << i << "  season_name: " << seasonName(season) << "\n"
             << i << "  time_of_day: " << timeOfDay << "\n"
             << i << "  clock: \"" << clock.str() << "\"\n"
             << i << "  background: " << background << "\n"
             << i << "  camera:\n"
             << i << "    position: " << vec(eye) << "\n"
             << i << "    target: " << vec(lookAt) << "\n"
             << i << "    direction: " << vec(glm::normalize(lookAt - eye)) << "\n"
             << i << "    up: [0, 1, 0]\n"
             << i << "    fov_y_degrees: " << kFov << "\n"
             << i << "    near: 0.05\n"
             << i << "    far: 60\n"
             << i << "    view: " << matrix(camera.view) << "\n"
             << i << "    projection: " << matrix(camera.proj) << "\n"
             << i << "  light:\n"
             << i << "    source: " << (sun ? "sun" : "moon") << "\n"
             << i << "    direction: " << vec(glm::vec3(s.sun)) << "\n"
             << i << "    elevation_degrees: " << glm::degrees(sun ? sky.sunElevation : sky.moonElevation) << "\n"
             << i << "    intensity: " << s.sun.w << "\n"
             << i << "    color: " << vec(glm::vec3(s.lightColor)) << "\n"
             << i << "  sun:\n"
             << i << "    direction: " << vec(sky.sunDirection) << "\n"
             << i << "    elevation_degrees: " << glm::degrees(sky.sunElevation) << "\n"
             << i << "  moon:\n"
             << i << "    direction: " << vec(sky.moonDirection) << "\n"
             << i << "    elevation_degrees: " << glm::degrees(sky.moonElevation) << "\n"
             << i << "  ambient_sky: " << vec(glm::vec3(s.sky)) << "\n"
             << i << "  ambient_ground: " << vec(glm::vec3(s.ground)) << "\n"
             << i << "  exposure: " << s.sky.w << "\n"
             << i << "  point_light:\n"
             << i << "    position: " << vec(glm::vec3(s.pointLight)) << "\n"
             << i << "    color: " << vec(glm::vec3(s.pointColor)) << "\n";
    };
    const std::string header =
        "# Renderings of the klartraum tech demo 001 bonsai scene, written by techdemo001_example --dataset.\n"
        "# World: units of the scene, +y up, +x east, -z north. Directions are unit vectors. Matrices are\n"
        "# row-major lists of 16 (view: camera from world, right-handed, looking along -z; projection:\n"
        "# Vulkan clip from camera, depth 0..1, y down in the image). Colours are linear rgb.\n"
        "# season: 0..4, spring 1.25, summer 2, fall 3.25, winter 4 = 0; time_of_day: 0..1, noon 0.5.\n"
        "# light: the directional light that casts the shadows, the sun by day and the full moon by night\n"
        "# (direction: towards it); ambient_sky/ambient_ground: ambient light from above/below; point_light:\n"
        "# the lantern's flame without shadows, colour = radiance at distance 1; exposure: of the tone curve.\n"
        "# classes: each surface's class as a colour (background black, pot [0.2, 0.4, 1], soil [0.6, 0.4, 0],\n"
        "# trunk and branches [1, 0, 0], needles [0, 1, 0], pumpkin [1, 0.6, 0], lantern [1, 1, 1]); depth: the\n"
        "# distance from the camera as grey from depth_near (black) to depth_far (white, also the background).\n";
    // A random camera around the scene's focus, from low to steep, near to far, looking at a point near the focus.
    const auto randomView = [&]() {
        const float azimuth = uniform(0.0f, 2.0f * 3.14159265f), elevation = uniform(0.1f, 0.95f);
        const float distance = uniform(5.0f, 9.0f);
        const glm::vec3 lookAt = scene.focus - glm::vec3(0.0f, 0.45f, 0.0f) +
                                 glm::vec3(uniform(-0.3f, 0.3f), uniform(-0.2f, 0.3f), uniform(-0.3f, 0.3f));
        return std::tuple{azimuth, elevation, distance, lookAt};
    };
    const auto eyeOf = [](float azimuth, float elevation, float distance, glm::vec3 lookAt) {
        return lookAt + distance * glm::vec3(std::cos(elevation) * std::sin(azimuth), std::sin(elevation),
                                             std::cos(elevation) * std::cos(azimuth));
    };

    std::ofstream yaml(directory / "dataset.yaml");
    yaml << std::setprecision(6) << header << "width: " << extent.width << "\nheight: " << extent.height
         << "\nseed: " << seed << "\ndepth_near: " << kDepthNear << "\ndepth_far: " << kDepthFar << "\nfield_box_low: ["
         << kFieldBoxLow.x << ", " << kFieldBoxLow.y << ", " << kFieldBoxLow.z << "]"
         << "\nfield_box_high: [" << kFieldBoxHigh.x << ", " << kFieldBoxHigh.y << ", " << kFieldBoxHigh.z << "]"
         << "\nimages:\n";
    for (uint32_t i = 0; i < count; ++i) {
        const auto [azimuth, elevation, distance, lookAt] = randomView();
        const float timeOfDay = uniform(0.0f, 1.0f), season = uniform(0.0f, 4.0f);
        std::ostringstream suffix;
        suffix << std::setw(4) << std::setfill('0') << i;
        shoot(yaml, "  ", directory, suffix.str(), eyeOf(azimuth, elevation, distance, lookAt), lookAt, timeOfDay,
              season, 0.37f * float(i));
        if ((i + 1) % 20 == 0)
            std::cout << "dataset: " << i + 1 << " of " << count << std::endl;
    }
    if (sequences > 0) {
        std::filesystem::create_directories(directory / "sequences");
        std::ofstream sequenceYaml(directory / "sequences" / "sequences.yaml");
        sequenceYaml << std::setprecision(6) << header << "width: " << extent.width << "\nheight: " << extent.height
                     << "\nseed: " << seed << "\ndepth_near: " << kDepthNear << "\ndepth_far: " << kDepthFar
                     << "\nframes_per_sequence: " << kSequenceFrames << "\nsequences:\n";
        for (uint32_t q = 0; q < sequences; ++q) {
            const auto [azimuth, elevation, distance, lookAt] = randomView();
            const float timeOfDay = uniform(0.0f, 1.0f), season = uniform(0.0f, 4.0f);
            // A slow orbit (0.6 degrees a frame) and three minutes of the day a frame, in either direction.
            const float turn = (uniform(0.0f, 1.0f) < 0.5f ? -1.0f : 1.0f) * 0.0105f;
            std::ostringstream name;
            name << "seq_" << std::setw(3) << std::setfill('0') << q;
            const std::filesystem::path folder = directory / "sequences" / name.str();
            std::filesystem::create_directories(folder);
            sequenceYaml << "  - name: " << name.str() << "\n    frames:\n";
            for (uint32_t f = 0; f < kSequenceFrames; ++f) {
                std::ostringstream suffix;
                suffix << std::setw(2) << std::setfill('0') << f;
                const float t = timeOfDay + 0.002f * float(f);
                shoot(sequenceYaml, "      ", folder, suffix.str(),
                      eyeOf(azimuth + turn * float(f), elevation, distance, lookAt), lookAt, t - std::floor(t), season,
                      0.37f * float(q) + float(f) / 30.0f);
            }
            if ((q + 1) % 10 == 0)
                std::cout << "sequences: " << q + 1 << " of " << sequences << std::endl;
        }
    }
    std::cout << "dataset in " << directory.string() << std::endl;
}

} // namespace

int main(int argc, char** argv) {
    std::filesystem::path potPath = "build/TestingOutput/splat_training/bonsai/object.spz";
    std::filesystem::path photosPath = "build/TestingOutput/real_data/pumpkin_closed_1_512";
    std::filesystem::path outDir = "build/TestingOutput/techdemo001";
    VkExtent2D extent{1280, 720};
    float fps = 30.0f;
    int fromChapter = 0, toChapter = int(kChapters.size()) - 1;
    uint32_t step = 1;
    uint32_t datasetCount = 0, sequenceCount = 0, datasetSeed = 1;
    std::filesystem::path stylePath, fieldPath, panelsPath;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const bool hasValue = i + 1 < argc;
        if (arg == "--pot" && hasValue)
            potPath = argv[++i];
        else if (arg == "--photos" && hasValue)
            photosPath = argv[++i];
        else if (arg == "--width" && hasValue)
            extent.width = uint32_t(std::stoul(argv[++i]));
        else if (arg == "--height" && hasValue)
            extent.height = uint32_t(std::stoul(argv[++i]));
        else if (arg == "--fps" && hasValue)
            fps = std::stof(argv[++i]);
        else if (arg == "--from" && hasValue)
            fromChapter = std::stoi(argv[++i]);
        else if (arg == "--to" && hasValue)
            toChapter = std::stoi(argv[++i]);
        else if (arg == "--step" && hasValue)
            step = std::max(1u, uint32_t(std::stoul(argv[++i])));
        else if (arg == "--out-dir" && hasValue)
            outDir = argv[++i];
        else if (arg == "--dataset" && hasValue)
            datasetCount = uint32_t(std::stoul(argv[++i]));
        else if (arg == "--seed" && hasValue)
            datasetSeed = uint32_t(std::stoul(argv[++i]));
        else if (arg == "--sequences" && hasValue)
            sequenceCount = uint32_t(std::stoul(argv[++i]));
        else if (arg == "--style" && hasValue)
            stylePath = argv[++i];
        else if (arg == "--field" && hasValue)
            fieldPath = argv[++i];
        else if (arg == "--panels" && hasValue)
            panelsPath = argv[++i];
    }
    if (datasetCount > 0) {
        // Only the dataset, of the scene with the pumpkin of an earlier run: no chapters.
        fromChapter = int(kChapters.size());
        toChapter = -1;
    }
    std::filesystem::create_directories(outDir / "frames");
    const float aspect = float(extent.width) / float(extent.height);

    HeadlessFrontend frontend;
    VulkanContext& vc = frontend.getKlartraumEngine().getVulkanContext();

    DemoTimeline timeline;
    timeline.fps = fps;
    for (const Chapter& chapter : kChapters)
        timeline.chapterSeconds.push_back(chapter.seconds);
    std::vector<Caption> captions;
    const auto unbounded = std::make_shared<Font>("fonts/Unbounded.ttf");
    const auto mono = std::make_shared<Font>("fonts/JetBrainsMono.ttf");
    for (const Chapter& chapter : kChapters)
        captions.push_back(makeCaption(chapter, unbounded, mono, extent.height));
    const auto frameName = [&](uint32_t frame) {
        std::ostringstream name;
        name << "frame_" << std::setw(4) << std::setfill('0') << frame << ".ppm";
        return outDir / "frames" / name.str();
    };
    // The sun as a black symbol by day and the moon as a white crescent by night, where they shine from, behind the
    // scene; each fades with the background and as it nears the horizon.
    const auto drawSky = [&](std::vector<uint8_t>& bgra, const glm::mat4& view, const TreeSky& sky, float background) {
        const float radius = 0.028f * float(extent.height);
        const SkyMark sun = skyMarkAt(view, sky.sunDirection, extent.width, extent.height);
        if (sun.visible)
            drawSunMark(bgra, extent.width, extent.height, sun.pixel, radius, glm::vec3(0.0f),
                        sun.opacity * background * smooth(sky.sunElevation / 0.08f), background);
        const SkyMark moon = skyMarkAt(view, sky.moonDirection, extent.width, extent.height);
        if (moon.visible)
            drawMoonMark(bgra, extent.width, extent.height, moon.pixel, radius,
                         litSideTowards(view, sky.moonDirection, sky.sunDirection), glm::vec3(1.0f),
                         moon.opacity * (1.0f - background) * smooth(sky.moonElevation / 0.08f), background);
    };
    const auto finishFrame = [&](uint32_t frame, std::vector<uint8_t> bgra, float background) {
        const int chapter = timeline.chapterAt(frame);
        const float opacity = timeline.captionOpacity({chapter, 0.6f, 0.4f, 0.5f}, frame);
        drawCaption(bgra, extent, captions[chapter], opacity, 1.0f - background);
        writePpm(frameName(frame), bgra, extent);
    };

    // The bonsai: the splatted pot with soil, lantern and a bonsai pine.
    BonsaiSceneConfig sceneConfig;
    sceneConfig.treeCount = 1;
    sceneConfig.species = BonsaiSpecies::Pine;
    sceneConfig.style = 1.0f;
    sceneConfig.pot.soilMound = 0.4f;
    sceneConfig.treeSize = 1.25f;
    sceneConfig.treeSpread = 0.65f;
    sceneConfig.potObject = std::make_shared<BonsaiObject>(loadBonsaiObject(potPath.string()));
    const BonsaiScene sceneWithoutPumpkin = makeBonsaiScene(sceneConfig);

    // The stage: the title hangs high above the pot, out of the pot's view; the camera comes down with the pumpkin.
    const glm::vec3 title = sceneWithoutPumpkin.focus + glm::vec3(0.0f, 5.0f, 0.0f);
    // Framed between the logo and the pumpkin above it, nearly level with the pumpkin.
    const Pose titlePose{title + glm::vec3(0.0f, 0.7f, 7.0f), title + glm::vec3(0.0f, 0.7f, 0.0f)};
    const float orbitElevation = 0.36f, orbitDistance = 7.6f, orbitStart = 0.35f;
    const auto orbitPose = [&](float angle) {
        // A little below the trees' focus, so that the pot stands in the frame.
        const glm::vec3 focus = sceneWithoutPumpkin.focus - glm::vec3(0.0f, 0.45f, 0.0f);
        return Pose{focus + orbitDistance * glm::vec3(std::cos(orbitElevation) * std::sin(angle),
                                                      std::sin(orbitElevation),
                                                      std::cos(orbitElevation) * std::cos(angle)),
                    focus};
    };
    const float morphScale = 0.35f; // the pumpkin's unit sphere in world units, at rest on the soil
    const float titleScale = 0.7f;  // the pumpkin's unit sphere in world units, above the title
    const glm::vec3 sphereCentre = title + glm::vec3(0.0f, 1.45f, 0.0f);
    // Where the jack-o'-lantern's face looks in the photographs' frame: -z turned by this about +y.
    const float faceAngle = 0.26f;
    // The rotation about +y that turns the face, once at -z, towards @p eye.
    const auto facing = [](glm::vec3 at, glm::vec3 eye) { return std::atan2(at.x - eye.x, at.z - eye.z); };
    // The pumpkin with its face turned to -z, then turned by `yaw` about +y and tipped by `tilt` towards its face
    // (its top towards -z), e.g. towards a camera below it, so that its unseen underside stays out of view.
    const auto pumpkinModel = [&](glm::vec3 at, float yaw, float scale, float tilt = 0.0f) {
        return glm::translate(glm::mat4(1.0f), at) * glm::rotate(glm::mat4(1.0f), yaw, glm::vec3(0.0f, 1.0f, 0.0f)) *
               glm::rotate(glm::mat4(1.0f), -tilt, glm::vec3(1.0f, 0.0f, 0.0f)) *
               glm::rotate(glm::mat4(1.0f), -faceAngle, glm::vec3(0.0f, 1.0f, 0.0f)) *
               glm::scale(glm::mat4(1.0f), glm::vec3(scale));
    };
    const float titleTilt = 0.45f; // radians, above the title; upright on the soil, which the camera sees from above
    const glm::vec2 landing(-0.75f, 0.35f);
    const float secondsPerTurn = 40.0f;
    const auto growthOrbit = [&](float seconds) { return orbitStart + 2.0f * 3.14159265f * seconds / secondsPerTurn; };

    // The pumpkin's photographs, masked, with their cameras in the unit sphere around it (up +y): the morph's targets.
    const raster::FittingScene photos = raster::loadFittingScene(photosPath);
    const glm::mat4 sceneFromUnit = photos.sceneFromUnitSphere();
    // All training photographs, glare included (a glossy sheen on the skin that one view sees and the others do not).
    const std::vector<uint32_t> used = photos.train;
    raster::MeshMorphOptions morphOptions;
    morphOptions.photographs = photos.targetsOf(used);
    for (const glm::mat4& camera : photos.camerasOf(used))
        morphOptions.photoCameras.push_back(camera * sceneFromUnit);
    std::cout << used.size() << " photographs of " << photos.width << " x " << photos.height << std::endl;
    morphOptions.training.stepsPerSubmit = 26;
    // The photographs' glare is a highlight the training learns: a normal map, an amplitude and the lights it reflects.
    morphOptions.training.specular = true;
    // A coarser material than the texture: the normal map follows the skin's ribs and bumps, not the photographs'
    // small misalignments.
    morphOptions.training.materialWidth = 256;
    morphOptions.training.materialHeight = 128;
    morphOptions.training.material.learningRate = 0.002f;
    // What the highlight cannot explain is let pass rather than bleaching the texture.
    morphOptions.training.highlightTolerance = 0.15f;
    morphOptions.training.width = photos.width;
    morphOptions.training.height = photos.height;
    // Fine enough for the carved face and the skin's ribs at the photographs' resolution.
    morphOptions.training.textureWidth = 1024;
    morphOptions.training.textureHeight = 512;
    morphOptions.frame = extent;

    // The trained pumpkin, carried from chapter to chapter. The sphere it starts from is turned so that its texture
    // seam (at -x, where the longitude wraps) lies at the back, opposite the face.
    raster::TriangleMesh pumpkinMesh = raster::icosphere(4);
    const glm::mat3 seamToBack(
        glm::rotate(glm::mat4(1.0f), 0.5f * 3.14159265f + faceAngle, glm::vec3(0.0f, 1.0f, 0.0f)));
    for (glm::vec3& p : pumpkinMesh.positions)
        p = seamToBack * p;
    std::vector<float> pumpkinTexture(
        4 * size_t(morphOptions.training.textureWidth) * morphOptions.training.textureHeight, 0.0f);
    for (size_t i = 3; i < pumpkinTexture.size(); i += 4)
        pumpkinTexture[i] = 1.0f; // black
    // Its specular reflection as the training learns it (MeshTraining::specularMaterial(), specularLights()).
    std::vector<float> pumpkinMaterial, pumpkinLights;

    const auto chapterRange = [&](int chapter) {
        return std::pair{timeline.chapterStart(chapter), chapter + 1 < int(kChapters.size())
                                                             ? timeline.chapterStart(chapter + 1)
                                                             : timeline.frameCount()};
    };
    std::cout << "tech demo 001: " << timeline.frameCount() << " frames at " << fps << " fps" << std::endl;

    // The trees at `age`: from nothing (0) through a seedling branching out step by step to fully grown (1).
    const auto setTrees = [](BonsaiRenderer& bonsai, const BonsaiScene& scene, float age) {
        std::vector<TreeInstance> trees = scene.trees;
        for (TreeInstance& tree : trees)
            setTreeAge(tree, age);
        bonsai.getGenerator().getInstances().set(trees);
    };
    // The morph's mesh as it is after each frame's steps casts the shadows of live occluder 0, each vertex standing for
    // its share of the start mesh's area.
    const auto castsShadows = [&](BonsaiRenderer& bonsai, raster::MeshMorph& morph) {
        float area = 0.0f;
        for (const glm::uvec3& t : pumpkinMesh.triangles)
            area += 0.5f * glm::length(glm::cross(pumpkinMesh.positions[t.y] - pumpkinMesh.positions[t.x],
                                                  pumpkinMesh.positions[t.z] - pumpkinMesh.positions[t.x]));
        const auto vertices = uint32_t(pumpkinMesh.positions.size());
        bonsai.connectLiveOccluder(0, BufferRef{morph.getTraining().getPositions(), -1}, vertices,
                                   area / float(vertices), morph.getTraining().getOutput());
    };
    // The light volume reaches up to the pumpkin above the title, so that it is lit and shadowed all the way down.
    const float lightCeiling = sphereCentre.y + titleScale + 0.3f;

    // Chapter 0: the title and the morph, on white.
    if (fromChapter <= 0 && toChapter >= 0) {
        TitleLayout layout;
        const float size = 3.2f / measureLine({"klartraum", unbounded, 500.0f, 1.0f, 0.0f});
        layout.lines.push_back({"klartraum", unbounded, 500.0f, size, 0.0f});
        layout.lines.push_back({"tech demo 001", mono, 400.0f, 0.4f * size, 0.05f});
        layout.pixelsPerUnit = 0.5f * float(extent.height) / (7.0f * std::tan(glm::radians(0.5f * kFov))) * 2.0f;
        TextSplatParams params;
        params.color = glm::vec3(0.0f);
        params.spacing = 0.006f;
        std::vector<Gaussian3D> text = sampleTextGaussians(rasterizeTitle(layout), params);
        for (Gaussian3D& g : text) {
            g.position[0] += title.x;
            g.position[1] += title.y - 0.45f;
            g.position[2] += title.z;
        }
        auto textData = std::make_shared<GaussianDataStandard>(vc, text);
        std::vector<glm::vec4> textColours;
        for (const Gaussian3D& g : text)
            textColours.emplace_back(g.color[0], g.color[1], g.color[2], g.alpha);

        auto frame = std::make_shared<OffscreenTarget>(vc, extent, 1);
        frame->setClearColor({{1.0f, 1.0f, 1.0f, 1.0f}});
        auto camera = std::make_shared<CameraUboType>();
        auto splatting = createGaussianSplatting(vc, GsplatBackend::Raster, frame, camera, textData->buffers());
        // The bonsai's light, high up to the title, lights the morph from the first frame on as it will be lit
        // later; the bonsai itself is not drawn (only its light volume is in the graph). The morph's mesh, as it
        // morphs, casts the shadows.
        BonsaiRenderConfig lightConfig;
        lightConfig.extent = VkExtent2D{64, 64};
        lightConfig.numberPaths = 1;
        lightConfig.lightCeiling = lightCeiling;
        lightConfig.liveOccluders = 1;
        BonsaiRenderer light(vc, sceneWithoutPumpkin, std::make_shared<OffscreenTarget>(vc, lightConfig.extent, 1),
                             lightConfig);
        raster::MeshMorphOptions litMorph = morphOptions;
        litMorph.lightSettings = light.getSettingsRef();
        litMorph.light = light.getLightVolume().getLight();
        raster::MeshMorph morph(vc, pumpkinMesh, GaussianSoABuffers{}, frame, splatting->getOutputLayout(), splatting,
                                litMorph);
        castsShadows(light, morph);
        ComputeGraph graph(vc, 1);
        // Not light.initialize(): it fills the buffers of the bonsai's drawing, which is not in this graph.
        graph.compileFrom(morph.getOutput());
        morph.initialize();
        morph.getTraining().setTexture(pumpkinTexture);
        camera->ubo = cameraOf(titlePose, aspect);
        camera->update(0);
        morph.setCamera(camera->ubo.proj * camera->ubo.view);
        const float titleYaw = facing(sphereCentre, titlePose.eye);
        const glm::mat4 model = pumpkinModel(sphereCentre, titleYaw, titleScale, titleTilt);
        morph.setModel(model);
        light.setLiveOccluder(0, model);
        light.setTime(0.0f, 0.5f, 2.0f, glm::vec4(0.0f));
        setTrees(light, sceneWithoutPumpkin, 0.0f);
        auto colours = std::dynamic_pointer_cast<BufferElementSinglePath<VulkanBuffer<glm::vec4>>>(
            textData->buffers().colAlpha.element);
        // On the right, from the start of the training on: the photograph the frame's last training step used and the
        // training's rendering of the mesh from that photograph's camera, at half their size, labelled; a new pair
        // every frame, so that the prediction is seen getting better from the sphere on.
        const uint32_t panelWidth = morphOptions.training.width / 2, panelHeight = morphOptions.training.height / 2;
        const auto halve = [&](const float* picture) {
            std::vector<float> half(4 * size_t(panelWidth) * panelHeight);
            const uint32_t width = morphOptions.training.width;
            for (uint32_t y = 0; y < panelHeight; ++y)
                for (uint32_t x = 0; x < panelWidth; ++x)
                    for (int c = 0; c < 4; ++c)
                        half[4 * (size_t(y) * panelWidth + x) + c] =
                            0.25f * (picture[4 * (size_t(2 * y) * width + 2 * x) + c] +
                                     picture[4 * (size_t(2 * y) * width + 2 * x + 1) + c] +
                                     picture[4 * (size_t(2 * y + 1) * width + 2 * x) + c] +
                                     picture[4 * (size_t(2 * y + 1) * width + 2 * x + 1) + c]);
            return half;
        };
        const auto label = [&](const char* text) {
            TitleLayout layout;
            layout.align = TextAlign::Left;
            layout.pixelsPerUnit = float(extent.height);
            layout.lines.push_back({text, mono, 400.0f, 0.021f, 0.0f});
            return rasterizeTitle(layout);
        };
        const CoverageMask photographLabel = label("photograph"), renderingLabel = label("rendering");
        std::vector<float> photographPanel, renderingPanel;
        const float trainingStart = 0.5f; // seconds into the title
        const auto drawTraining = [&](std::vector<uint8_t>& bgra, raster::MeshMorph& morph, float opacity) {
            if (opacity <= 0.0f)
                return;
            {
                const uint32_t view = morph.getTraining().lastView();
                const size_t pixels = size_t(morphOptions.training.width) * morphOptions.training.height;
                photographPanel = halve(&morphOptions.photographs[4 * pixels * view]);
                std::vector<float> rendering(morph.getTraining().getRendering()->getDataElementCount());
                morph.getTraining().getRendering()->getDataBuffer(0).memcopyTo(rendering);
                renderingPanel = halve(rendering.data());
            }
            const int x = int(extent.width) - int(panelWidth) - int(0.05f * float(extent.width));
            const int gap = int(0.03f * float(extent.height));
            const int total = 2 * int(panelHeight) + 2 * int(photographLabel.height) + 3 * gap;
            int y = (int(extent.height) - total) / 2;
            for (const auto& [picture, text] :
                 {std::pair{&photographPanel, &photographLabel}, {&renderingPanel, &renderingLabel}}) {
                blendCaption(bgra, extent.width, extent.height, *text, x, y, glm::vec3(0.0f), opacity);
                y += int(text->height) + gap / 2;
                blendPicture(bgra, extent.width, extent.height, *picture, panelWidth, panelHeight, x, y, opacity);
                y += int(panelHeight) + gap;
            }
        };
        const auto [begin, end] = chapterRange(0);
        for (uint32_t f = begin; f < end; ++f) {
            // The title fades out over the chapter's last second.
            const float fade = smooth((timeline.timeInChapter(end - 1) - timeline.timeInChapter(f)) / 1.0f);
            std::vector<glm::vec4> faded = textColours;
            for (glm::vec4& c : faded)
                c.a *= fade;
            colours->getBuffer().memcopyFrom(faded);
            // The title holds with the black sphere for its first half second: the training starts over before each
            // of those frames, so that it begins when the panels appear.
            const float sinceStart = timeline.timeInChapter(f);
            if (sinceStart < trainingStart) {
                morph.initialize();
                morph.getTraining().setTexture(pumpkinTexture);
            }
            graph.submitAndWait(vc.getGraphicsQueue(), 0);
            if ((f - begin) % step == 0) {
                std::vector<uint8_t> image = readImage(vc, frame->getImage(0), splatting->getOutputLayout(), extent);
                // From the start of the training on, fading in, and out with the title.
                drawTraining(image, morph, fade * smooth((sinceStart - trainingStart) / 0.3f));
                finishFrame(f, std::move(image), 1.0f);
                std::cout << "frame " << f << ": morph loss " << morph.getTraining().loss() << std::endl;
            }
        }
        pumpkinMesh = morph.getTraining().mesh();
        pumpkinTexture = morph.getTraining().texture();
        pumpkinMaterial = morph.getTraining().specularMaterial();
        pumpkinLights = morph.getTraining().specularLights();
        raster::writeObj(pumpkinMesh, outDir / "pumpkin.obj", "pumpkin.ppm");
        raster::writeTexturePpm(pumpkinTexture, morphOptions.training.textureWidth, morphOptions.training.textureHeight,
                                outDir / "pumpkin.ppm");
        writeFloats(outDir / "pumpkin_material.bin", pumpkinMaterial);
        writeFloats(outDir / "pumpkin_lights.bin", pumpkinLights);
    } else {
        // Later chapters only: the pumpkin of an earlier run.
        std::filesystem::path texturePath;
        pumpkinMesh = raster::readObj(outDir / "pumpkin.obj", &texturePath);
        uint32_t w = 0, h = 0;
        pumpkinTexture = raster::readTexturePpm(texturePath, w, h);
        pumpkinMaterial = readFloats(outDir / "pumpkin_material.bin");
        pumpkinLights = readFloats(outDir / "pumpkin_lights.bin");
    }

    // Where the pumpkin comes to rest: on the soil, at its trained size, glossy as the photographs showed it.
    BonsaiObjectPlacement placement;
    BonsaiSpecular gloss;
    const size_t materialCount = size_t(morphOptions.training.materialWidth) * morphOptions.training.materialHeight;
    if (pumpkinMaterial.size() == 4 * materialCount && pumpkinLights.size() == 4 * (raster::kSpecularLights + 1)) {
        gloss.material = pumpkinMaterial;
        gloss.width = morphOptions.training.materialWidth;
        gloss.height = morphOptions.training.materialHeight;
        gloss.exponent = 6.0f + std::exp(pumpkinLights[4 * raster::kSpecularLights]);
        gloss.brightness = 0.0f;
        for (uint32_t k = 0; k < raster::kSpecularLights; ++k)
            gloss.brightness = std::max(gloss.brightness, pumpkinLights[4 * k + 3] * pumpkinLights[4 * k + 3]);
    }
    placement.object = std::make_shared<BonsaiObject>(
        bonsaiObjectFromMesh(pumpkinMesh, pumpkinTexture, morphOptions.training.textureWidth,
                             morphOptions.training.textureHeight, gloss.material.empty() ? nullptr : &gloss));
    placement.position = landing;
    // Facing the camera on the soil; placement.yaw turns the photographs' frame, whose face is turned by faceAngle.
    const float restYaw = facing(glm::vec3(landing.x, 0.0f, landing.y), orbitPose(orbitStart).eye);
    placement.yaw = restYaw - faceAngle;
    placement.height = morphScale * (placement.object->high.y - placement.object->low.y);
    const glm::mat4 rest = bonsaiObjectTransform(sceneWithoutPumpkin, placement);
    const glm::vec3 restOrigin(rest[3]);

    BonsaiRenderConfig renderConfig;
    renderConfig.extent = extent;
    renderConfig.mode = BonsaiRenderMode::Classical;
    renderConfig.background = true;
    renderConfig.numberPaths = 1;

    // Chapter 1: the drop, the pumpkin still drawn by the morph (which keeps refining it), lit by the scene and
    // casting its shadow as it falls.
    if (fromChapter <= 1 && toChapter >= 1) {
        auto frame = std::make_shared<OffscreenTarget>(vc, extent, 1);
        BonsaiRenderConfig dropConfig = renderConfig;
        dropConfig.lightCeiling = lightCeiling;
        dropConfig.liveOccluders = 1;
        BonsaiRenderer bonsai(vc, sceneWithoutPumpkin, frame, dropConfig);
        const VkImageLayout layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        raster::MeshMorphOptions litMorph = morphOptions;
        litMorph.lightSettings = bonsai.getSettingsRef();
        litMorph.light = bonsai.getLightVolume().getLight();
        raster::MeshMorph morph(vc, pumpkinMesh, GaussianSoABuffers{}, frame, layout, bonsai.getOutput(), litMorph);
        castsShadows(bonsai, morph);
        ComputeGraph graph(vc, 1);
        graph.compileFrom(morph.getOutput());
        bonsai.initialize(1);
        morph.initialize();
        morph.getTraining().setTexture(pumpkinTexture);
        bonsai.setBackground(glm::vec3(1.0f));
        const auto [begin, end] = chapterRange(1);
        const float duration = float(end - begin) / fps;
        for (uint32_t f = begin; f < end; ++f) {
            const float t = timeline.timeInChapter(f);
            // The camera comes down from the title to the pot; the pumpkin falls with it, lands and bounces once.
            const Pose pose = mix(titlePose, orbitPose(orbitStart), smooth(t / duration));
            const float fall = std::min(t / (0.75f * duration), 1.0f);
            glm::vec3 origin = glm::mix(sphereCentre, restOrigin, fall * fall);
            if (fall >= 1.0f) {
                const float after = (t - 0.75f * duration) / (0.25f * duration);
                origin.y += 0.12f * placement.height * std::sin(3.14159265f * std::clamp(after, 0.0f, 1.0f));
            }
            bonsai.getCamera()->ubo = cameraOf(pose, aspect);
            bonsai.getCamera()->update(0);
            bonsai.setView(pose.eye);
            bonsai.setTime(t, 0.5f, 2.0f, glm::vec4(0.0f));
            setTrees(bonsai, sceneWithoutPumpkin, 0.0f);
            morph.setCamera(bonsai.getCamera()->ubo.proj * bonsai.getCamera()->ubo.view);
            const float settle = smooth(fall);
            // The short way round from the title's yaw to the resting one.
            const float titleYaw = facing(sphereCentre, titlePose.eye);
            const float turn = std::remainder(restYaw - titleYaw, 2.0f * 3.14159265f);
            const glm::mat4 model = pumpkinModel(origin, titleYaw + settle * turn,
                                                 glm::mix(titleScale, morphScale, settle), (1.0f - settle) * titleTilt);
            morph.setModel(model);
            bonsai.setLiveOccluder(0, model);
            graph.submitAndWait(vc.getGraphicsQueue(), 0);
            if ((f - begin) % step == 0) {
                finishFrame(f, readImage(vc, frame->getImage(0), layout, extent), 1.0f);
                std::cout << "frame " << f << std::endl;
            }
        }
    }

    // Chapters 2 and 3: the pumpkin rests in the scene; the trees grow, then the days pass.
    if (toChapter >= 2) {
        BonsaiSceneConfig withPumpkin = sceneConfig;
        withPumpkin.objects.push_back(placement);
        const BonsaiScene scene = makeBonsaiScene(withPumpkin);
        auto frame = std::make_shared<OffscreenTarget>(vc, extent, 1);
        BonsaiRenderer bonsai(vc, scene, frame, renderConfig);
        // With --style, the days in the watercolor style: the network restyles every frame into `styled`.
        // A network that reads the G-buffers gets them from two more renders of each frame, copied aside.
        std::unique_ptr<StyleTransfer> style;
        auto styled = std::make_shared<OffscreenTarget>(vc, extent, 1);
        auto classes = std::make_shared<OffscreenTarget>(vc, extent, 1);
        auto depth = std::make_shared<OffscreenTarget>(vc, extent, 1);
        // Filled by copies between submissions: the graph must not clear them.
        classes->setClear(false);
        depth->setClear(false);
        if (!stylePath.empty())
            style = std::make_unique<StyleTransfer>(vc, stylePath.string(), frame, styled, bonsai.getOutput(), classes,
                                                    depth);
        // With --field, the style anchored in the world (FieldStyle): it reads the classes, the coverage and the
        // depth in its box's window, each from one more render of every frame.
        std::unique_ptr<FieldStyle> field;
        auto coverage = std::make_shared<OffscreenTarget>(vc, extent, 1);
        coverage->setClear(false);
        if (!fieldPath.empty())
            field = std::make_unique<FieldStyle>(vc, fieldPath.string() + ".onnx", fieldPath.string() + ".bin", frame,
                                                 classes, coverage, depth, styled, bonsai.getOutput());
        ComputeGraph graph(vc, 1);
        graph.compileFrom(field ? field->getOutput() : style ? style->getOutput() : bonsai.getOutput());
        // With --panels, the style's training data beside the days.
        std::unique_ptr<StylePanels> panels;
        if (!panelsPath.empty())
            panels = std::make_unique<StylePanels>(panelsPath, extent, mono);
        bonsai.initialize(1);
        BonsaiDays days;
        // The growth's summer noon turns into the night over the first seconds; then four days from midnight.
        // First the style's demonstration: the summer afternoon holds while the style fades in; then the afternoon
        // turns into the night, and four days from midnight.
        const float demonstration = kChapters[3].seconds, afternoon = 4.0f;
        days.secondsPerDay = (kChapters[4].seconds - afternoon) / 4.0f;
        days.dawn = 0.0f;
        days.seasons[0] = 2.0f, days.seasons[1] = 3.25f, days.seasons[2] = 4.0f, days.seasons[3] = 5.25f;
        const uint32_t begin = timeline.chapterStart(std::max(fromChapter, 2)), end = timeline.frameCount();
        const uint32_t growthStart = timeline.chapterStart(2), daysStart = timeline.chapterStart(3);
        // The time the wind's gusts follow: faster in storms, integrated so that it never jumps.
        float windTime = 0.0f;
        for (uint32_t f = begin; f < end; ++f) {
            const float sinceGrowth = float(f - growthStart) / fps;
            const Pose pose = orbitPose(growthOrbit(sinceGrowth));
            bonsai.getCamera()->ubo = cameraOf(pose, aspect);
            if (panels && f >= daysStart) {
                // The bonsai moves left of the panels over the first second of the days (a shift of the image's
                // centre, so the view itself stays the same).
                const float shift = -0.36f * smooth(float(f - daysStart) / fps);
                bonsai.getCamera()->ubo.proj =
                    glm::translate(glm::mat4(1.0f), glm::vec3(shift, 0.0f, 0.0f)) * bonsai.getCamera()->ubo.proj;
            }
            bonsai.getCamera()->update(0);
            bonsai.setView(pose.eye);
            float background = 1.0f, timeOfDay = 0.5f, season = 2.0f;
            if (f < daysStart) {
                // Quick at first, slowing down as the tree matures.
                const float t = std::clamp(sinceGrowth / kChapters[2].seconds, 0.0f, 1.0f);
                const float growth = 1.0f - (1.0f - t) * (1.0f - t);
                setTrees(bonsai, scene, 0.02f + 0.98f * growth);
                bonsai.setTime(sinceGrowth, timeOfDay, season, glm::vec4(1.0f, 0.0f, 0.0f, 0.3f * growth));
            } else {
                // The camera keeps orbiting.
                const float seconds = std::max(float(f - daysStart) / fps - demonstration, 0.0f);
                setTrees(bonsai, scene, 1.0f);
                timeOfDay =
                    seconds < afternoon ? 0.5f + 0.5f * seconds / afternoon : days.timeOfDayAt(seconds - afternoon);
                season = seconds < afternoon ? days.seasons[0] : days.seasonAt(seconds - afternoon);
                // Storms in fall and winter: from summer (2) the wind rises into fall (3.25), blows through winter
                // (4) and calms down towards spring (5.25).
                const float storm = smooth((season - 2.5f) / 0.6f) * (1.0f - smooth((season - 4.5f) / 0.6f));
                windTime += (1.0f + 1.5f * storm) / fps;
                bonsai.setTime(sinceGrowth + windTime, timeOfDay, season,
                               glm::vec4(1.0f, 0.0f, 0.0f, 0.4f + 1.8f * storm));
                background = backgroundAt(timeOfDay, season);
            }
            bonsai.setBackground(glm::vec3(background));
            // The watercolor style fades in over two seconds, one second into the days.
            const float sinceDays = float(f) / fps - float(daysStart) / fps;
            const float blend = f < daysStart ? 0.0f : smooth((sinceDays - 1.0f) / 2.0f);
            if (style) {
                style->setBlend(blend);
                if (style->usesGBuffers() && blend > 0.0f) {
                    bonsai.setOutput(BonsaiOutput::Classes);
                    bonsai.setBackground(glm::vec3(0.0f));
                    graph.submitAndWait(vc.getGraphicsQueue(), 0);
                    copyImage(vc, frame->getImage(0), classes->getImage(0), extent);
                    bonsai.setOutput(BonsaiOutput::Depth, kDepthNear, kDepthFar);
                    bonsai.setBackground(glm::vec3(1.0f));
                    graph.submitAndWait(vc.getGraphicsQueue(), 0);
                    copyImage(vc, frame->getImage(0), depth->getImage(0), extent);
                    bonsai.setOutput(BonsaiOutput::Colour);
                    bonsai.setBackground(glm::vec3(background));
                }
            }
            if (field) {
                field->setBlend(blend);
                const auto [near, far] = field->depthWindow(pose.eye);
                const TreeRenderSettings settings = bonsai.getSettings().values().at(0);
                const TreeSky sky = treeSkyAt(timeOfDay, season);
                const bool sun = glm::dot(sky.lightDirection, sky.sunDirection) > 0.999f;
                const CameraMVP& camera = bonsai.getCamera()->ubo;
                field->setFrame(camera.proj * camera.view, pose.eye, near, far,
                                fieldConditions(timeOfDay, season, settings.sun.w, sun, settings.sky.w, background));
                if (blend > 0.0f) {
                    const std::pair<std::shared_ptr<OffscreenTarget>, std::function<void()>> gbuffers[] = {
                        {classes, [&] { bonsai.setOutput(BonsaiOutput::Classes); }},
                        {coverage, [&] { bonsai.setOutput(BonsaiOutput::Depth, -2.0f, -1.0f); }},
                        {depth, [&] { bonsai.setOutput(BonsaiOutput::Depth, near, far); }}};
                    bonsai.setBackground(glm::vec3(0.0f));
                    for (const auto& [target, select] : gbuffers) {
                        select();
                        graph.submitAndWait(vc.getGraphicsQueue(), 0);
                        copyImage(vc, frame->getImage(0), target->getImage(0), extent);
                    }
                    bonsai.setOutput(BonsaiOutput::Colour);
                    bonsai.setBackground(glm::vec3(background));
                }
            }
            graph.submitAndWait(vc.getGraphicsQueue(), 0);
            if ((f - begin) % step == 0) {
                std::vector<uint8_t> image = readImage(vc, (style || field ? styled : frame)->getImage(0),
                                                       VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, extent);
                if (kSkyMarks)
                    drawSky(image, bonsai.getCamera()->ubo.view, treeSkyAt(timeOfDay, season), background);
                if (panels && f >= daysStart)
                    panels->draw(image, sinceDays, demonstration, season, background);
                finishFrame(f, std::move(image), background);
                std::cout << "frame " << f << std::endl;
            }
        }
    }
    if (datasetCount > 0) {
        BonsaiSceneConfig withPumpkin = sceneConfig;
        withPumpkin.objects.push_back(placement);
        const BonsaiScene scene = makeBonsaiScene(withPumpkin);
        writeDataset(vc, scene, renderConfig, extent, datasetCount, sequenceCount, datasetSeed, outDir / "dataset");
        return 0;
    }
    std::cout << "frames in " << (outDir / "frames").string() << std::endl;
    return 0;
}
