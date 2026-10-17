// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Renders "klartraum tech demo 000", a promo video in one continuous shot
// around the lantern scene, headlessly. The storyboard has six chapters; this
// renderer implements:
//   1. Title: "klartraum" / "tech demo 000", made of the lantern's own
//      Gaussians, holds still and morphs into the scene (swirl-scatter)
//   2. Gaussian splatting: the captured lantern, with a slow push-in
//   3. Dynamic lighting: dusk falls, a candle flickers in the lantern's fire
//      box and a moon light circles the scene (GaussianLighting)
//   4. Neural character animation: the mixture-of-experts controller walks a
//      plain skeleton, drawn as Gaussians (--mesh-skeleton: as rasterized
//      cylinders and spheres), in from the side and around the lantern
//   5. OpenPose keypoints: the skeleton turns into the OpenPose pose map
//   6. Diffusion on the pose: SD1.5 + ControlNet-OpenPose + LCM-LoRA inpaint a
//      person over the skeleton (about 8 s per frame), with the pose map
//      ControlNet sees close up, picture in picture; --placeholder shows the
//      character mask instead, to work on the cinematics quickly
//   7. Finale: the scene's Gaussians explode and the picture fades to black
// Each chapter shows a caption with the papers it builds on, lower left.
//
//   techdemo000_example [--from 1] [--to 7] [--step 1] [--placeholder] [--mesh-skeleton] [--prompt TEXT] [--seed 5]
//                       [--backend raster|compute] [--sd-path gpu|cpu|both] [--dump-graphs DIR]
//                       [--width 1920] [--height 1080] [--fps 30]
//                       [--seconds-per-turn 15] [--file data/lantern.spz]
//                       [--out-dir build/TestingOutput/techdemo000]
//
// --backend selects the Gaussian splatting backend of every graph (default:
// raster). --sd-path selects how chapter 6 generates the person: entirely on
// the GPU as one chain of graphs per frame (gpu, the default), with the CPU
// between the graphs (cpu), or both, writing the GPU path's frames to
// <out-dir>/frames_gpu and their difference and times to <out-dir>/sd_compare.csv.
// --dump-graphs records GPU timestamps in the graphs of the GPU path and writes them, with the mean time of each
// element over the generated frames, as JSON at the end. At the end, the mean time of the
// per-frame graph submission is printed per chapter.
//
// Frames go to <out-dir>/frames/frame_XXXX.ppm, numbered through the whole
// video, so a range of chapters can be re-rendered into the same sequence.
// The program checks the frames it renders: the title frames must be
// identical, and the first frame after the morph must equal the scene rendered
// directly from the same camera. It exits with an error otherwise.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "klartraum/character_controller.hpp"
#include "klartraum/character_mask.hpp"
#include "klartraum/computegraph/buffertographicsbarrier.hpp"
#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/gaussianmerge.hpp"
#include "klartraum/computegraph/gaussianmorph.hpp"
#include "klartraum/computegraph/graph_export.hpp"
#include "klartraum/computegraph/graphchain.hpp"
#include "klartraum/computegraph/hostvalues.hpp"
#include "klartraum/computegraph/imageviewsrc.hpp"
#include "klartraum/computegraph/renderpass.hpp"
#include "klartraum/computegraph/skeletongaussians.hpp"
#include "klartraum/demo_timeline.hpp"
#include "klartraum/gaussian_data_standard.hpp"
#include "klartraum/gaussian_lighting.hpp"
#include "klartraum/gaussian_morph.hpp"
#include "klartraum/gaussian_splatting_factory.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/interface_camera_orbit.hpp"
#include "klartraum/offscreen_target.hpp"
#include "klartraum/onnx/onnx_network.hpp"
#include "klartraum/openpose_skeleton_renderer.hpp"
#include "klartraum/scene_placement.hpp"
#include "klartraum/sd15/closeup_inpainting.hpp"
#include "klartraum/sd15/controlnet_denoiser.hpp"
#include "klartraum/sd15/gpu_inpainting.hpp"
#include "klartraum/sd15/latent_inpainting.hpp"
#include "klartraum/sd15/lcm_scheduler.hpp"
#include "klartraum/sd15/pose_video_pipeline.hpp"
#include "klartraum/skeleton_mesh_renderer.hpp"
#include "klartraum/text_gaussians.hpp"

namespace {

// A chapter of the storyboard: its length and its caption (a heading, a line
// on what happens, the papers).
struct Chapter {
    const char* name;
    float seconds;
    float captionDelay; // seconds after the chapter start
    std::string heading;
    std::string body;
    std::vector<std::string> references;
};

// Chapters without a heading show no caption, without a body no line under the heading.
const std::vector<Chapter> kChapters = {
    {"title", 3.5f, 0.0f, "", "", {}},
    {"gaussian_splatting",
     7.5f,
     0.5f,
     "3D Gaussian splatting",
     "Rendered by a Vulkan-based single-dispatch compute and render graph.",
     {"Kerbl, Kopanas, Leimkühler, Drettakis - 3D Gaussian Splatting for Real-Time Radiance Field Rendering.",
      "SIGGRAPH 2023.", "Zwicker, Pfister, van Baar, Gross - EWA Volume Splatting. IEEE Visualization 2001."}},
    {"dynamic_lighting",
     10.0f,
     0.5f,
     "Dynamic lighting",
     "Per-splat normals and baked SH visibility, lighting by up to " +
         std::to_string(klartraum::LightingFrame::kMaxLights) + " dynamic light sources.",
     {"Gao et al. - Relightable 3D Gaussians: Realistic Point Cloud Relighting with BRDF Decomposition",
      "and Ray Tracing. ECCV 2024.",
      "Sloan, Kautz, Snyder - Precomputed Radiance Transfer for Real-Time Rendering in Dynamic,",
      "Low-Frequency Lighting Environments. SIGGRAPH 2002."}},
    {"character",
     10.0f,
     0.5f,
     "Neural character animation",
     "A mixture-of-experts controller running in the Vulkan-based compute graph.",
     {"Zhang, Starke, Komura, Saito - Mode-Adaptive Neural Networks for Quadruped Motion Control. SIGGRAPH 2018.",
      "Training data: 100STYLE - Mason, Starke, Komura. I3D 2022."}},
    {"openpose",
     5.0f,
     0.5f,
     "OpenPose keypoints",
     "(you know what is coming).",
     {"Cao, Hidalgo, Simon, Wei, Sheikh - OpenPose: Realtime Multi-Person 2D Pose Estimation using Part",
      "Affinity Fields. IEEE TPAMI 43(1), 2021."}},
    {"diffusion",
     13.0f,
     0.5f,
     "Diffusion on the pose",
     "SD1.5 steered by ControlNet inpaints the character within the scene geometry.",
     {"Rombach, Blattmann, Lorenz, Esser, Ommer - High-Resolution Image Synthesis with Latent Diffusion",
      "Models. CVPR 2022.",
      "Zhang, Rao, Agrawala - Adding Conditional Control to Text-to-Image Diffusion Models. ICCV 2023.",
      "Luo et al. - LCM-LoRA: A Universal Stable-Diffusion Acceleration Module. arXiv 2311.05556, 2023."}},
    {"finale", 4.0f, 0.0f, "", "", {}},
};

// Reads an OffscreenTarget image back as tightly-packed BGRA bytes. The
// backends leave an OffscreenTarget in TRANSFER_SRC_OPTIMAL.
std::vector<uint8_t> readImage(klartraum::VulkanContext& vc, VkImage image, VkExtent2D extent,
                               VkImageLayout layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
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
    vkCmdCopyImageToBuffer(cmd, image, layout, buf, 1, &region);
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

// A caption rasterized once: one mask per text style, stacked from the top.
struct CaptionBlock {
    struct Part {
        klartraum::CoverageMask mask;
        glm::vec3 colour;
        int top; // offset from the top of the block, pixels
    };
    std::vector<Part> parts;
    int width = 0;
    int height = 0;
};

CaptionBlock makeCaption(const Chapter& chapter, const std::shared_ptr<klartraum::Font>& headingFont,
                         const std::shared_ptr<klartraum::Font>& bodyFont, uint32_t imageHeight) {
    // Sizes scale with the image; these are for 1080 lines.
    const float scale = float(imageHeight) / 1080.0f;
    CaptionBlock block;
    auto add = [&](const std::vector<std::string>& lines, const std::shared_ptr<klartraum::Font>& font, float weight,
                   float size, const glm::vec3& colour, float gapBefore) {
        klartraum::TitleLayout layout;
        layout.align = klartraum::TextAlign::Left;
        layout.pixelsPerUnit = 1.0f; // sizes in pixels
        layout.lineSpacing = 1.35f;
        for (const std::string& line : lines) {
            layout.lines.push_back({line, font, weight, size * scale, 0.0f});
        }
        const int top = block.height + int(gapBefore * scale);
        block.parts.push_back({klartraum::rasterizeTitle(layout), colour, top});
        block.height = top + block.parts.back().mask.height;
        block.width = std::max(block.width, block.parts.back().mask.width);
    };
    if (chapter.heading.empty()) {
        return block;
    }
    add({chapter.heading}, headingFont, 500.0f, 40.0f, glm::vec3(1.0f), 0.0f);
    if (!chapter.body.empty()) {
        add({chapter.body}, bodyFont, 400.0f, 22.0f, glm::vec3(1.0f), 14.0f);
    }
    if (!chapter.references.empty()) {
        add(chapter.references, bodyFont, 400.0f, 17.0f, glm::vec3(0.72f), 18.0f);
    }
    return block;
}

float smoothstep(float edge0, float edge1, float x) {
    const float t = std::clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// Darkens the lower-left corner behind a caption block, fully up to the
// block's right and top edges and fading out over `fade` pixels beyond them.
void darkenBehind(std::vector<uint8_t>& frame, VkExtent2D extent, int right, int top, int fade, float strength) {
    const int x1 = std::min(int(extent.width), right + fade);
    for (int y = std::max(0, top - fade); y < int(extent.height); ++y) {
        const float wy = 1.0f - smoothstep(float(top), float(top - fade), float(y));
        for (int x = 0; x < x1; ++x) {
            const float wx = 1.0f - smoothstep(float(right), float(right + fade), float(x));
            const float keep = 1.0f - strength * wx * wy;
            uint8_t* p = &frame[(size_t(y) * extent.width + x) * 4];
            for (int c = 0; c < 3; ++c) {
                p[c] = uint8_t(float(p[c]) * keep + 0.5f);
            }
        }
    }
}

// Draws the caption with its bottom-left corner `margin` pixels from the
// image's, over a darkened corner and with a soft shadow under the text, so
// it stays readable over the scene.
void drawCaption(std::vector<uint8_t>& frame, VkExtent2D extent, const CaptionBlock& block, float opacity) {
    if (opacity <= 0.0f || block.parts.empty()) {
        return;
    }
    const int margin = int(72.0f * float(extent.height) / 1080.0f);
    const int left = margin;
    const int top = int(extent.height) - margin - block.height;
    darkenBehind(frame, extent, left + block.width, top, margin * 2, 0.6f * opacity);
    for (const auto& part : block.parts) {
        klartraum::blendCaption(frame, extent.width, extent.height, part.mask, left + 2, top + part.top + 2,
                                glm::vec3(0.0f), 0.6f * opacity);
    }
    for (const auto& part : block.parts) {
        klartraum::blendCaption(frame, extent.width, extent.height, part.mask, left, top + part.top, part.colour,
                                opacity);
    }
}

constexpr int kLightingChapter = 2;
// Height of the candle in the fire box and the radius of its glow, relative
// to the lantern's height.
constexpr float kFireBoxHeight = 0.6f;
constexpr float kGlowRadius = 0.2f;

float easeInOut(float s) {
    s = std::clamp(s, 0.0f, 1.0f);
    return s < 0.5f ? 4.0f * s * s * s : 1.0f - std::pow(-2.0f * s + 2.0f, 3.0f) / 2.0f;
}

// Rises from 0 to 1 between a and b.
float ramp(float t, float a, float b) { return easeInOut((t - a) / (b - a)); }

// The lights of chapter 3 at `t` seconds into it: the relit look blends in
// over 1.5 s and dusk falls; a candle in the fire box flickers for a few
// seconds; then a red, a green and a blue light circle the lantern once; in
// the last 1.5 s the scene fades back to its captured look.
klartraum::LightingFrame chapterLighting(float t, float length, const glm::vec3& camera,
                                         const klartraum::UprightObjectBounds& lantern) {
    klartraum::LightingFrame frame;
    frame.camera = glm::vec4(camera, 0.0f);
    frame.control.x = ramp(t, 0.0f, 1.5f) * (1.0f - ramp(t, length - 1.5f, length));
    const float dusk = ramp(t, 1.0f, 3.0f) * (1.0f - ramp(t, length - 2.5f, length - 1.0f));
    frame.ambient = glm::vec4(glm::mix(glm::vec3(1.0f), glm::vec3(0.2f, 0.22f, 0.3f), dusk), 1.0f);

    const float candleOn = ramp(t, 1.5f, 2.5f) * (1.0f - ramp(t, 4.0f, 5.0f));
    const float flicker = 1.0f + 0.12f * std::sin(2.0f * 3.14159f * 7.3f * t) +
                          0.08f * std::sin(2.0f * 3.14159f * 13.1f * t + 1.0f) +
                          0.05f * std::sin(2.0f * 3.14159f * 2.7f * t + 2.0f);
    const glm::vec3 fireBox = lantern.base + glm::vec3(0.0f, kFireBoxHeight * lantern.height, 0.0f);
    const float reference = 0.35f * lantern.height;
    // The candle shines out of the fire box's windows, which the coarse
    // visibility closes, so it is unshadowed; the fire box itself glows.
    frame.lights[0] = {glm::vec4(fireBox, 0.0f), glm::vec4(1.0f, 0.55f, 0.22f, 3.5f * candleOn * flicker),
                       glm::vec4(reference, 0.15f * reference, 1.0f, kGlowRadius * lantern.height)};

    // Three coloured lights, a third of a turn apart, circle the lantern at
    // half its height, once between 4.5 s and 8.5 s.
    const float colourOn = ramp(t, 4.5f, 5.5f) * (1.0f - ramp(t, 7.5f, 8.5f));
    const float circleAngle = 2.0f * 3.14159f * std::clamp((t - 4.5f) / 4.0f, 0.0f, 1.0f);
    const float circleRadius = 0.9f * lantern.groundRadius;
    const glm::vec3 colours[3] = {{1.0f, 0.12f, 0.08f}, {0.1f, 1.0f, 0.15f}, {0.12f, 0.25f, 1.0f}};
    for (int k = 0; k < 3; ++k) {
        const float angle = circleAngle + float(k) * 2.0f * 3.14159f / 3.0f;
        const glm::vec3 position = lantern.base + glm::vec3(circleRadius * std::cos(angle), 0.5f * lantern.height,
                                                            circleRadius * std::sin(angle));
        frame.lights[1 + k] = {glm::vec4(position, 0.0f), glm::vec4(colours[k], 2.0f * colourOn),
                               glm::vec4(circleRadius, 0.1f * circleRadius, 0.0f, 0.0f)};
    }
    frame.counts.x = 4;
    return frame;
}

constexpr int kCharacterChapter = 3;
constexpr int kOpenPoseChapter = 4;
constexpr int kDiffusionChapter = 5;
constexpr int kFinaleChapter = 6;
constexpr uint32_t kGenerationSize = 512; // SD1.5 close-up, pixels
// The finale's explosion: how long the burst takes, how far the Gaussians fly
// (relative to the lantern's height) and when the fade to black starts.
constexpr float kExplosionSeconds = 3.0f;
constexpr float kExplosionRadius = 1.5f;
constexpr float kFadeStart = 1.5f;
constexpr float kCharacterHeight = 1.75f; // metres, the controller's skeleton
constexpr float kWalkSpeed = 1.2f;        // metres per second
constexpr float kRunSpeed = 3.5f;         // on the approach, before slowing to a walk

// Blends the pose map over the frame where it has a limb or joint.
void overlayPose(std::vector<uint8_t>& frame, const std::vector<uint8_t>& pose, float opacity) {
    if (opacity <= 0.0f) {
        return;
    }
    for (size_t i = 0; i < frame.size(); i += 4) {
        if (pose[i] == 0 && pose[i + 1] == 0 && pose[i + 2] == 0) {
            continue;
        }
        for (int c = 0; c < 3; ++c) {
            frame[i + c] = uint8_t(float(frame[i + c]) + (float(pose[i + c]) - float(frame[i + c])) * opacity + 0.5f);
        }
    }
}

// The square of the image around a character: centred on its joints, as wide
// as `factor` times its height on screen (projected through `mvp`).
struct CropSquare {
    float x0 = 0.0f, y0 = 0.0f, size = 1.0f;
};

CropSquare characterCrop(const std::vector<glm::vec4>& joints, const glm::mat4& mvp, VkExtent2D extent, float factor) {
    glm::vec2 low(1e30f), high(-1e30f), sum(0.0f);
    int count = 0;
    for (const glm::vec4& joint : joints) {
        const glm::vec4 clip = mvp * glm::vec4(glm::vec3(joint), 1.0f);
        if (clip.w <= 0.0f) {
            continue;
        }
        const glm::vec2 pixel((clip.x / clip.w * 0.5f + 0.5f) * float(extent.width),
                              (clip.y / clip.w * 0.5f + 0.5f) * float(extent.height));
        low = glm::min(low, pixel);
        high = glm::max(high, pixel);
        sum += pixel;
        ++count;
    }
    if (count == 0) {
        return {};
    }
    const glm::vec2 centre = sum / float(count);
    const float size = std::max(factor * (high.y - low.y), 8.0f);
    return {centre.x - 0.5f * size, centre.y - 0.5f * size, size};
}

// Draws `crop` of the pose map as a square of `side` pixels in the upper right
// corner, with a thin grey frame and `label` below it, `opacity` blended over
// the frame.
void drawPictureInPicture(std::vector<uint8_t>& frame, VkExtent2D extent, const std::vector<uint8_t>& pose,
                          VkExtent2D poseExtent, const CropSquare& crop, uint32_t side,
                          const klartraum::CoverageMask& label, float opacity) {
    if (opacity <= 0.0f) {
        return;
    }
    const uint32_t margin = extent.height / 15;
    const uint32_t x0 = extent.width - margin - side, y0 = margin;
    for (uint32_t y = 0; y < side; ++y) {
        for (uint32_t x = 0; x < side; ++x) {
            const bool border = x < 2 || y < 2 || x + 2 >= side || y + 2 >= side;
            // Nearest pixel of the crop; outside the image is black.
            const int sx = int(crop.x0 + (float(x) + 0.5f) * crop.size / float(side));
            const int sy = int(crop.y0 + (float(y) + 0.5f) * crop.size / float(side));
            const bool inside = sx >= 0 && sy >= 0 && sx < int(poseExtent.width) && sy < int(poseExtent.height);
            uint8_t* q = &frame[(size_t(y0 + y) * extent.width + x0 + x) * 4];
            for (int c = 0; c < 3; ++c) {
                float value = 160.0f;
                if (!border) {
                    value = inside ? float(pose[(size_t(sy) * poseExtent.width + sx) * 4 + c]) : 0.0f;
                }
                q[c] = uint8_t(float(q[c]) + (value - float(q[c])) * opacity + 0.5f);
            }
        }
    }
    klartraum::blendCaption(frame, extent.width, extent.height, label, int(x0), int(y0 + side + margin / 4),
                            glm::vec3(0.72f), opacity);
}

// Fills the character mask with a light grey, the placeholder for the
// generated person, `opacity` blended over the frame.
void overlayMask(std::vector<uint8_t>& frame, const std::vector<uint8_t>& mask, float opacity) {
    if (opacity <= 0.0f) {
        return;
    }
    for (size_t i = 0; i < frame.size(); i += 4) {
        const float a = float(mask[i + 2]) / 255.0f * opacity;
        if (a <= 0.0f) {
            continue;
        }
        for (int c = 0; c < 3; ++c) {
            frame[i + c] = uint8_t(float(frame[i + c]) + (210.0f - float(frame[i + c])) * a + 0.5f);
        }
    }
}

constexpr float kVaeScalingFactor = 0.18215f;
constexpr VkBufferUsageFlags kTensorUsage =
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
constexpr VkImageLayout kRendered = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

std::filesystem::path controlDirectory(uint32_t size) { return "data/onnx/sd15_controlnet_" + std::to_string(size); }
std::filesystem::path vaeDirectory(uint32_t size) { return "data/onnx/sd15_" + std::to_string(size); }

/**
 * The close-up around the character, rendered through the crop cameras: the scene with its depth, the pose map, the
 * character mask with a margin and without one (the body, the region pasted into the scene).
 */
struct CloseUpRender {
    std::shared_ptr<klartraum::OffscreenTarget> sceneTarget, poseTarget, maskTarget, pasteTarget;
    std::shared_ptr<klartraum::ComputeGraphElement> splatting;
    std::shared_ptr<klartraum::RenderPass> posePass, maskPass, pastePass;
};

CloseUpRender renderCloseUp(klartraum::VulkanContext& vc, VkExtent2D extent, klartraum::GsplatBackend backend,
                            std::shared_ptr<klartraum::GaussianDataStandard> model,
                            std::shared_ptr<klartraum::BufferElementInterface> joints,
                            std::shared_ptr<klartraum::CameraUboType> sceneCamera,
                            std::shared_ptr<klartraum::CameraUboType> skeletonCamera) {
    CloseUpRender r;
    r.sceneTarget = std::make_shared<klartraum::OffscreenTarget>(vc, extent, 1u);
    klartraum::GsplatConfig config;
    config.outputDepth = true;
    r.splatting = klartraum::createGaussianSplatting(vc, backend, r.sceneTarget, sceneCamera, model, config);
    const auto depth = klartraum::getGaussianSplattingDepth(r.splatting);
    auto pass = [&](std::shared_ptr<klartraum::OffscreenTarget>& target,
                    std::shared_ptr<klartraum::DrawComponent> component, klartraum::ComputeGraphElementPtr after) {
        target = std::make_shared<klartraum::OffscreenTarget>(vc, extent, 1u);
        auto renderPass = std::make_shared<klartraum::RenderPass>(vc.getSwapChainImageFormat(), extent);
        renderPass->setInput(target, 0);
        renderPass->setInput(skeletonCamera, 1);
        renderPass->addDrawComponent(component);
        renderPass->addComputeDependency(after);
        return renderPass;
    };
    auto skeleton = std::make_shared<klartraum::OpenPoseSkeletonRenderer>(joints, 1, extent);
    skeleton->setOcclusionDepth(depth);
    r.posePass = pass(r.poseTarget, skeleton, joints);
    r.posePass->addComputeDependency(r.splatting);
    auto mask = std::make_shared<klartraum::CharacterMaskRenderer>(joints, extent);
    mask->setOcclusionDepth(depth);
    r.maskPass = pass(r.maskTarget, mask, r.posePass);
    auto pasteMask = std::make_shared<klartraum::CharacterMaskRenderer>(joints, extent);
    pasteMask->setMargin(0.0f);
    pasteMask->setOcclusionDepth(depth);
    r.pastePass = pass(r.pasteTarget, pasteMask, r.maskPass);
    return r;
}

/// What a person generator returns for a frame.
struct GeneratedFrame {
    std::vector<uint8_t> full;        ///< the scene with the person, in the full view (BGRA)
    std::vector<uint8_t> closeUpPose; ///< the close-up pose map ControlNet saw (BGRA, size x size)
};

/**
 * Inpaints a person into the scene with SD1.5, ControlNet-OpenPose and LCM-LoRA, as lantern_walk_example does,
 * for a full view of any aspect ratio, with the CPU between the graphs.
 *
 * The person is small in the full view, too small for SD1.5. Every frame is therefore generated in a square
 * close-up of the generation size (see closeup_inpainting.hpp): a crop camera zooms onto a window around the
 * character, and a graph of its own renders the scene with its depth, the pose map and the character mask through
 * it. The close-up is inpainted by latent blending (every step keeps the scene's latent outside the mask) and pasted
 * into the full frame inside the full view's character mask.
 */
class PersonGenerator {
public:
    PersonGenerator(klartraum::VulkanContext& vc, uint32_t size, VkExtent2D fullExtent,
                    std::shared_ptr<klartraum::GaussianDataStandard> model,
                    std::shared_ptr<klartraum::BufferElementInterface> joints, const glm::mat4& characterToScene,
                    const std::vector<float>& embedding, const std::vector<std::vector<float>>& noise,
                    klartraum::GsplatBackend backend)
        : vc(vc),
          extent{size, size},
          fullExtent(fullExtent),
          latentSize(size / 8),
          characterToScene(characterToScene),
          scheduler(klartraum::LcmScheduler::load(controlDirectory(size) / "lcm_schedule_f32.bin")),
          noise(noise) {
        sceneCamera = std::make_shared<klartraum::CameraUboType>();
        skeletonCamera = std::make_shared<klartraum::CameraUboType>();
        closeUp = renderCloseUp(vc, extent, backend, model, joints, sceneCamera, skeletonCamera);
        closeUpGraph = std::make_unique<klartraum::ComputeGraph>(vc, 1);
        closeUpGraph->compileFrom(closeUp.pastePass);

        encoder = vc.create<klartraum::OnnxNetwork>((vaeDirectory(size) / "sd15_vae_encoder.onnx").string());
        encoderInput =
            vc.create<klartraum::TensorElement<float>>(std::vector<uint32_t>{1, 3, size, size}, kTensorUsage);
        encoder->setInputTensor("input", encoderInput);
        encodeGraph = std::make_unique<klartraum::ComputeGraph>(vc, 1);
        encodeGraph->compileFrom(encoder);

        denoiser = std::make_shared<klartraum::ControlNetDenoiser>(vc, controlDirectory(size), latentSize);
        denoiseGraph = std::make_unique<klartraum::ComputeGraph>(vc, 1);
        denoiseGraph->compileFrom(denoiser);
        denoiser->getEmbeddings()->setData(0, embedding);

        decoder = vc.create<klartraum::OnnxNetwork>((vaeDirectory(size) / "sd15_vae_decoder.onnx").string());
        decoderInput = vc.create<klartraum::TensorElement<float>>(std::vector<uint32_t>{1, 4, latentSize, latentSize},
                                                                  kTensorUsage);
        decoder->setInputTensor("input", decoderInput);
        decodeGraph = std::make_unique<klartraum::ComputeGraph>(vc, 1);
        decodeGraph->compileFrom(decoder);
    }

    /**
     * The frame for the character's current joints. `scene` and `characterMask` are the full view's scene (without
     * the character) and character mask, seen with the camera `matrices`; `openPoseJoints` the joints in the
     * character's world.
     */
    GeneratedFrame frame(const std::vector<uint8_t>& scene, const std::vector<uint8_t>& characterMask,
                         const std::vector<glm::vec4>& openPoseJoints, const klartraum::CameraMVP& matrices) {
        window.update(openPoseJoints, matrices, characterToScene, float(fullExtent.width) / float(fullExtent.height));
        sceneCamera->ubo = window.sceneCamera(matrices);
        skeletonCamera->ubo = window.skeletonCamera(matrices, characterToScene);
        sceneCamera->update(0);
        skeletonCamera->update(0);
        closeUpGraph->submitAndWait(vc.getGraphicsQueue(), 0);
        const auto closeUpScene = readImage(vc, closeUp.sceneTarget->getImage(0), extent, kRendered);
        const auto closeUpPose = readImage(vc, closeUp.poseTarget->getImage(0), extent, kRendered);
        const auto closeUpMask =
            klartraum::redChannel(readImage(vc, closeUp.maskTarget->getImage(0), extent, kRendered));
        const auto pasteMask =
            klartraum::redChannel(readImage(vc, closeUp.pasteTarget->getImage(0), extent, kRendered));
        GeneratedFrame result{scene, closeUpPose};
        if (std::none_of(closeUpMask.begin(), closeUpMask.end(), [](uint8_t value) { return value != 0; })) {
            return result; // the character is hidden
        }

        const size_t pixels = size_t(extent.width) * extent.height;
        encoderInput->setData(0, klartraum::bgraToTensor(closeUpScene, pixels, -1.0f, 1.0f));
        encodeGraph->submitAndWait(vc.getGraphicsQueue(), 0);
        std::vector<float> known(4 * size_t(latentSize) * latentSize);
        std::dynamic_pointer_cast<klartraum::TensorElement<float>>(encoder->getOutputElement("output"))
            ->getDataBuffer(0)
            .memcopyTo(known);
        for (auto& value : known) {
            value *= kVaeScalingFactor;
        }

        denoiser->getPose()->setData(0, klartraum::bgraToTensor(closeUpPose, pixels, 0.0f, 1.0f));
        // Generated freely where the body is; in the margin ring around it only half, so SD stays close to the scene
        // there but clothing can still reach into it.
        const klartraum::LatentInpainting inpainting{
            klartraum::closeUpLatentMask(closeUpMask, pasteMask, extent.width, extent.height), known};
        auto latents = klartraum::sampleInpainted(
            scheduler, noise, inpainting, [&](const std::vector<float>& sample, int64_t timestep) {
                denoiser->getSample()->setData(0, sample);
                denoiser->getTimestep()->setData(0, std::vector<int64_t>{timestep});
                denoiseGraph->submitAndWait(vc.getGraphicsQueue(), 0);
                std::vector<float> prediction(known.size());
                denoiser->getNoisePrediction()->getDataBuffer(0).memcopyTo(prediction);
                return prediction;
            });
        for (auto& value : latents) {
            value /= kVaeScalingFactor;
        }
        decoderInput->setData(0, latents);
        decodeGraph->submitAndWait(vc.getGraphicsQueue(), 0);
        std::vector<float> generated(3 * pixels);
        std::dynamic_pointer_cast<klartraum::TensorElement<float>>(decoder->getOutputElement("output"))
            ->getDataBuffer(0)
            .memcopyTo(generated);

        const int size = int(extent.width);
        const auto pasted = klartraum::pasteGenerated(closeUpScene, generated, pasteMask, size);
        result.full = klartraum::compositeCloseUp(scene, characterMask, pasted, size, int(fullExtent.width),
                                                  int(fullExtent.height), window);
        return result;
    }

private:
    klartraum::VulkanContext& vc;
    VkExtent2D extent;
    VkExtent2D fullExtent;
    uint32_t latentSize;
    glm::mat4 characterToScene;
    klartraum::LcmScheduler scheduler;
    std::vector<std::vector<float>> noise;
    klartraum::CloseUpWindow window;

    std::shared_ptr<klartraum::CameraUboType> sceneCamera, skeletonCamera;
    CloseUpRender closeUp;
    std::unique_ptr<klartraum::ComputeGraph> closeUpGraph;
    std::shared_ptr<klartraum::OnnxNetwork> encoder;
    std::shared_ptr<klartraum::TensorElement<float>> encoderInput;
    std::unique_ptr<klartraum::ComputeGraph> encodeGraph;
    std::shared_ptr<klartraum::ControlNetDenoiser> denoiser;
    std::unique_ptr<klartraum::ComputeGraph> denoiseGraph;
    std::shared_ptr<klartraum::OnnxNetwork> decoder;
    std::shared_ptr<klartraum::TensorElement<float>> decoderInput;
    std::unique_ptr<klartraum::ComputeGraph> decodeGraph;
};

/**
 * PersonGenerator entirely on the GPU: the frame is one GraphChain, submitted at once,
 *
 *     prepare → [denoise → LCM step 0] → … → [denoise → LCM step 3] → decode
 *
 * - prepare: the close-up window and its cameras from the joints and the full view's camera, the close-up render,
 *   the close-up and the pose map as tensors, the latent mask, the VAE encoder and the start of LCM sampling;
 * - denoise: ControlNet + UNet, compiled once, four times in the chain;
 * - LCM step i: the scheduler step and latent blending, in place on the denoiser's sample;
 * - decode: the VAE decoder, the paste into the close-up and the composite into the full view.
 *
 * It reads the full view's scene and character mask, the camera and the joints where the main graph left them on
 * the GPU, so the main graph must have finished. The CPU reads back the finished frame and the close-up pose map.
 */
class GpuPersonGenerator {
public:
    GpuPersonGenerator(klartraum::VulkanContext& vc, uint32_t size, VkExtent2D fullExtent,
                       std::shared_ptr<klartraum::GaussianDataStandard> model,
                       std::shared_ptr<klartraum::BufferElementInterface> joints, const glm::mat4& characterToScene,
                       const std::vector<float>& embedding, const std::vector<std::vector<float>>& noise,
                       klartraum::GsplatBackend backend, std::shared_ptr<klartraum::CameraUboType> fullCamera,
                       const std::shared_ptr<klartraum::OffscreenTarget>& fullScene,
                       const std::shared_ptr<klartraum::OffscreenTarget>& fullMask, bool profile = false)
        : vc(vc),
          extent{size, size},
          fullExtent(fullExtent),
          model(model),
          profiling(profile),
          chain(vc) {
        // Profiling marks every element's commands with timestamps, which the graph pictures need (--dump-graphs).
        auto makeGraph = [&](bool simultaneous = false) {
            auto graph = std::make_unique<klartraum::ComputeGraph>(vc, 1);
            if (profile) {
                graph->enableProfiling();
            }
            if (simultaneous) {
                graph->enableSimultaneousUse();
            }
            return graph;
        };
        const uint32_t latentSize = size / 8;
        const auto scheduler = klartraum::LcmScheduler::load(controlDirectory(size) / "lcm_schedule_f32.bin");

        // prepare
        sceneCamera = std::make_shared<klartraum::CameraUboType>();
        sceneCamera->setName("close-up camera");
        auto skeletonCamera = std::make_shared<klartraum::CameraUboType>();
        skeletonCamera->setName("close-up camera");
        fullCamera->setName("camera");
        joints->setName("joints");
        window = klartraum::createCloseUpWindow(
            vc, joints, klartraum::CharacterController::kOpenPoseJointCount, fullCamera, characterToScene,
            float(fullExtent.width) / float(fullExtent.height), sceneCamera, skeletonCamera);
        closeUp = renderCloseUp(vc, extent, backend, model, joints, sceneCamera, skeletonCamera);
        closeUp.sceneTarget->setName("close-up");
        closeUp.poseTarget->setName("pose map");
        closeUp.maskTarget->setName("mask");
        closeUp.pasteTarget->setName("body mask");
        closeUp.posePass->setName("PosePass");
        closeUp.maskPass->setName("MaskPass");
        closeUp.pastePass->setName("BodyMaskPass");
        auto encoderInput =
            vc.create<klartraum::TensorElement<float>>(std::vector<uint32_t>{1, 3, size, size}, kTensorUsage);
        encoderInput->setName("close-up tensor");
        sceneToTensor =
            klartraum::createImageToTensor(vc, closeUp.sceneTarget, kRendered, encoderInput, extent, -1.0f, 1.0f);
        sceneToTensor->addDependency(closeUp.splatting);
        denoiser = std::make_shared<klartraum::ControlNetDenoiser>(vc, controlDirectory(size), latentSize);
        denoiser->getSample()->setName("latents");
        denoiser->getTimestep()->setName("timestep");
        denoiser->getEmbeddings()->setName("prompt embedding");
        denoiser->getPose()->setName("pose tensor");
        denoiser->getNoisePrediction()->setName("noise prediction");
        sceneToTensor->setName("SceneToTensor");
        auto poseToTensor =
            klartraum::createImageToTensor(vc, closeUp.poseTarget, kRendered, denoiser->getPose(), extent, 0.0f, 1.0f);
        poseToTensor->addDependency(closeUp.posePass);
        poseToTensor->setName("PoseToTensor");
        buffers = klartraum::LcmInpaintingBuffers::create(vc, 4, latentSize * latentSize, noise);
        auto latentMask = klartraum::createCloseUpLatentMask(vc, closeUp.maskTarget, kRendered, closeUp.pasteTarget,
                                                             kRendered, buffers.mask, extent);
        latentMask->addDependency(closeUp.pastePass);
        encoder = vc.create<klartraum::OnnxNetwork>((vaeDirectory(size) / "sd15_vae_encoder.onnx").string());
        encoder->setName("VAE encoder");
        encoder->setInputTensor("input", sceneToTensor, 1);
        auto init = klartraum::createLcmInit(vc, scheduler, kVaeScalingFactor, encoder->getOutputElement("output"),
                                             buffers, denoiser->getSample(), denoiser->getTimestep());
        init->addDependency(encoder);
        init->addDependency(latentMask);
        init->addDependency(poseToTensor);
        prepareGraph = makeGraph();
        prepareGraph->compileFrom(init);

        // denoise
        denoiseGraph = makeGraph(true);
        denoiseGraph->compileFrom(denoiser);
        denoiser->getEmbeddings()->setData(0, embedding);

        // LCM steps
        decoderInput = vc.create<klartraum::TensorElement<float>>(std::vector<uint32_t>{1, 4, latentSize, latentSize},
                                                                  kTensorUsage);
        decoderInput->setName("decoder input");
        for (size_t index = 0; index < scheduler.size(); ++index) {
            stepGraphs.push_back(makeGraph());
            stepGraphs.back()->compileFrom(klartraum::createLcmStep(
                vc, scheduler, index, kVaeScalingFactor, denoiser->getSample(), denoiser->getNoisePrediction(), buffers,
                denoiser->getTimestep(), decoderInput));
        }

        // decode. A target recorded in a graph clears its image, so the images rendered by other graphs (the
        // close-up's, the main graph's) are read through views that do not.
        auto wrap = [](const std::shared_ptr<klartraum::OffscreenTarget>& target, VkExtent2D imageExtent) {
            auto image = std::make_shared<klartraum::ImageViewSrc>(std::vector<VkImageView>{target->getImageView(0)},
                                                                   std::vector<VkImage>{target->getImage(0)},
                                                                   std::vector<VkExtent2D>{imageExtent});
            image->setClear(false);
            image->setName(target->getName());
            return image;
        };
        decoder = vc.create<klartraum::OnnxNetwork>((vaeDirectory(size) / "sd15_vae_decoder.onnx").string());
        decoder->setName("VAE decoder");
        decoder->setInputTensor("input", decoderInput);
        closeUpPixels = std::make_shared<klartraum::BufferElement<klartraum::VulkanBuffer<uint32_t>>>(vc, size * size);
        closeUpPixels->setName("pasted close-up");
        // The scene and the paste mask were read by prepare, which left them in GENERAL.
        auto paste = klartraum::createCloseUpPaste(
            vc, decoder->getOutputElement("output"), wrap(closeUp.sceneTarget, extent), VK_IMAGE_LAYOUT_GENERAL,
            wrap(closeUp.pasteTarget, extent), VK_IMAGE_LAYOUT_GENERAL, closeUpPixels, size);
        paste->addDependency(decoder);
        fullFrame = std::make_shared<klartraum::BufferElement<klartraum::VulkanBuffer<uint32_t>>>(
            vc, fullExtent.width * fullExtent.height);
        fullFrame->setName("frame with person");
        auto fullSceneView = wrap(fullScene, fullExtent);
        fullSceneView->setName("frame");
        auto fullMaskView = wrap(fullMask, fullExtent);
        fullMaskView->setName("character mask");
        auto composite = klartraum::createCloseUpComposite(vc, fullSceneView, kRendered, fullMaskView, kRendered,
                                                           closeUpPixels, window.state, fullFrame, fullExtent, size);
        composite->addDependency(paste);
        decodeGraph = makeGraph();
        decodeGraph->compileFrom(composite);

        chain.add(*prepareGraph);
        for (auto& step : stepGraphs) {
            chain.add(*denoiseGraph);
            chain.add(*step);
        }
        chain.add(*decodeGraph);
    }

    /// The frame for the joints and camera of the main graph's last submission.
    GeneratedFrame frame() {
        const auto start = std::chrono::steady_clock::now();
        chain.submit(vc.getGraphicsQueue());
        const auto submitted = std::chrono::steady_clock::now();
        chain.wait();
        chainMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        submitMs += std::chrono::duration<double, std::milli>(submitted - start).count();
        ++chainRuns;
        if (profiling) {
            // The denoiser's timestamps are those of its last run in the chain; all runs do the same work.
            prepareGraph->readProfiling();
            denoiseGraph->readProfiling();
            for (auto& step : stepGraphs) {
                step->readProfiling();
            }
            decodeGraph->readProfiling();
        }
        GeneratedFrame result;
        std::vector<uint32_t> packed(size_t(fullExtent.width) * fullExtent.height);
        fullFrame->getBuffer(0).memcopyTo(packed);
        result.full.resize(4 * packed.size());
        std::memcpy(result.full.data(), packed.data(), result.full.size());
        // The pose map was read into the pose tensor, which left it in GENERAL.
        result.closeUpPose = readImage(vc, closeUp.poseTarget->getImage(0), extent, VK_IMAGE_LAYOUT_GENERAL);
        return result;
    }

    /**
     * Writes the graphs of the chain as JSON to `directory` (klartraum::writeGraphJson()), in the order they were
     * compiled, and `chain.json` with the order of the links; scripts/techdemo000 draws them.
     */
    void exportGraphs(const std::filesystem::path& directory) const {
        std::filesystem::create_directories(directory);
        std::vector<klartraum::ComputeGraphElementPtr> splattingInputs{sceneCamera, closeUp.sceneTarget};
        for (const auto* ref : model->buffers().all()) {
            splattingInputs.push_back(ref->element);
        }
        const std::vector<klartraum::ComputeGraphElementPtr> denoiserInputs{
            denoiser->getSample(), denoiser->getTimestep(), denoiser->getEmbeddings(), denoiser->getPose()};
        auto unetInputs = denoiserInputs;
        unetInputs.push_back(denoiser->getControlNet());
        std::vector<std::string> compiled = {"prepare", "denoise"};
        klartraum::writeGraphJson(
            directory / "prepare.json", "prepare", *prepareGraph,
            {{"Gaussian splatting", closeUp.splatting, splattingInputs}, {"VAE encoder", encoder, {sceneToTensor}}});
        klartraum::writeGraphJson(
            directory / "denoise.json", "denoise", *denoiseGraph,
            {{"ControlNet", denoiser->getControlNet(), denoiserInputs}, {"UNet", denoiser->getUnet(), unetInputs}});
        for (size_t i = 0; i < stepGraphs.size(); ++i) {
            const std::string name = "lcm_step_" + std::to_string(i);
            klartraum::writeGraphJson(directory / (name + ".json"), name, *stepGraphs[i], {});
            compiled.push_back(name);
        }
        klartraum::writeGraphJson(directory / "decode.json", "decode", *decodeGraph,
                                  {{"VAE decoder", decoder, {decoderInput}}});
        compiled.push_back("decode");
        std::ofstream out(directory / "chain.json");
        out << "{\n  \"compiled\": [";
        for (size_t i = 0; i < compiled.size(); ++i) {
            out << (i ? ", " : "") << "\"" << compiled[i] << "\"";
        }
        out << "],\n  \"links\": [\"prepare\"";
        for (size_t i = 0; i < stepGraphs.size(); ++i) {
            out << ", \"denoise\", \"lcm_step_" << i << "\"";
        }
        const uint32_t runs = std::max(1u, chainRuns);
        out << ", \"decode\"],\n  \"wall_ms\": " << chainMs / runs << ",\n  \"submit_ms\": " << submitMs / runs;
        // Per compiled graph, the GPU time from its first element's start to its last element's end (the denoiser's
        // of its last run in the chain).
        out << ",\n  \"gpu_span_ms\": {\"prepare\": " << prepareGraph->getGpuSpanMs()
            << ", \"denoise\": " << denoiseGraph->getGpuSpanMs();
        for (size_t i = 0; i < stepGraphs.size(); ++i) {
            out << ", \"lcm_step_" << i << "\": " << stepGraphs[i]->getGpuSpanMs();
        }
        out << ", \"decode\": " << decodeGraph->getGpuSpanMs() << "}\n}\n";
    }

private:
    klartraum::VulkanContext& vc;
    VkExtent2D extent;
    VkExtent2D fullExtent;
    std::shared_ptr<klartraum::GaussianDataStandard> model;
    bool profiling;
    // From the submission of the chain until its fence, and the vkQueueSubmit call alone, summed over the frames.
    double chainMs = 0.0;
    double submitMs = 0.0;
    uint32_t chainRuns = 0;
    std::shared_ptr<klartraum::CameraUboType> sceneCamera;
    klartraum::CloseUpWindowElements window;
    CloseUpRender closeUp;
    std::shared_ptr<klartraum::ComputeGraphElement> sceneToTensor;
    std::shared_ptr<klartraum::TensorElement<float>> decoderInput;
    klartraum::LcmInpaintingBuffers buffers;
    std::shared_ptr<klartraum::OnnxNetwork> encoder;
    std::shared_ptr<klartraum::ControlNetDenoiser> denoiser;
    std::shared_ptr<klartraum::OnnxNetwork> decoder;
    std::shared_ptr<klartraum::BufferElement<klartraum::VulkanBuffer<uint32_t>>> closeUpPixels;
    std::shared_ptr<klartraum::BufferElement<klartraum::VulkanBuffer<uint32_t>>> fullFrame;
    std::unique_ptr<klartraum::ComputeGraph> prepareGraph;
    std::unique_ptr<klartraum::ComputeGraph> denoiseGraph;
    std::vector<std::unique_ptr<klartraum::ComputeGraph>> stepGraphs;
    std::unique_ptr<klartraum::ComputeGraph> decodeGraph;
    klartraum::GraphChain chain;
};

std::shared_ptr<klartraum::Font> loadFont(const std::string& name) {
    return std::make_shared<klartraum::Font>("fonts/" + name + ".ttf");
}

} // namespace

int main(int argc, char** argv) {
    int fromChapter = 1;
    int toChapter = int(kChapters.size());
    uint32_t step = 1;
    bool placeholder = false;
    bool meshSkeleton = false;
    klartraum::GsplatBackend backend = klartraum::GsplatBackend::Raster;
    std::string sdPath = "gpu";
    std::filesystem::path dumpGraphs;
    std::string prompt = "a photograph of a woman in a dark blue kimono walking in a japanese moss garden, full body, "
                         "natural light, realistic";
    uint32_t seed = 5;
    uint32_t width = 1920;
    uint32_t height = 1080;
    std::string spzFile = "./data/lantern.spz";
    std::filesystem::path outDir = "build/TestingOutput/techdemo000";
    klartraum::DemoTimeline timeline;
    timeline.secondsPerTurn = 15.0f;
    timeline.startDistance = 0.65f;
    timeline.endDistance = 0.55f;
    timeline.pushInChapter = 1;
    // Once the character appears, the camera turns four times slower, so it
    // can be followed.
    timeline.slowChapter = kCharacterChapter;
    timeline.slowSecondsPerTurn = 60.0f;
    for (const Chapter& chapter : kChapters) {
        timeline.chapterSeconds.push_back(chapter.seconds);
    }

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const bool hasValue = i + 1 < argc;
        if (arg == "--from" && hasValue) {
            fromChapter = std::stoi(argv[++i]);
        } else if (arg == "--to" && hasValue) {
            toChapter = std::stoi(argv[++i]);
        } else if (arg == "--placeholder") {
            placeholder = true;
        } else if (arg == "--backend" && hasValue) {
            const std::string name = argv[++i];
            if (name != "compute" && name != "raster") {
                std::cerr << "--backend must be compute or raster" << std::endl;
                return 1;
            }
            backend = name == "raster" ? klartraum::GsplatBackend::Raster : klartraum::GsplatBackend::Compute;
        } else if (arg == "--sd-path" && hasValue) {
            sdPath = argv[++i];
            if (sdPath != "cpu" && sdPath != "gpu" && sdPath != "both") {
                std::cerr << "--sd-path must be cpu, gpu or both" << std::endl;
                return 1;
            }
        } else if (arg == "--dump-graphs" && hasValue) {
            dumpGraphs = argv[++i];
        } else if (arg == "--mesh-skeleton") {
            meshSkeleton = true;
        } else if (arg == "--prompt" && hasValue) {
            prompt = argv[++i];
        } else if (arg == "--seed" && hasValue) {
            seed = uint32_t(std::stoul(argv[++i]));
        } else if (arg == "--step" && hasValue) {
            step = std::max(1u, uint32_t(std::stoul(argv[++i])));
        } else if (arg == "--width" && hasValue) {
            width = uint32_t(std::stoul(argv[++i]));
        } else if (arg == "--height" && hasValue) {
            height = uint32_t(std::stoul(argv[++i]));
        } else if (arg == "--fps" && hasValue) {
            timeline.fps = std::stof(argv[++i]);
        } else if (arg == "--seconds-per-turn" && hasValue) {
            timeline.secondsPerTurn = std::stof(argv[++i]);
        } else if (arg == "--file" && hasValue) {
            spzFile = argv[++i];
        } else if (arg == "--out-dir" && hasValue) {
            outDir = argv[++i];
        } else {
            std::cerr << "unknown or incomplete argument: " << arg << std::endl;
            return 1;
        }
    }
    if (fromChapter < 1 || toChapter > int(kChapters.size()) || fromChapter > toChapter) {
        std::cerr << "--from and --to must be chapters between 1 and " << kChapters.size() << std::endl;
        return 1;
    }
    if (width == 0 || height == 0 || !(timeline.fps > 0.0f) || !(timeline.secondsPerTurn > 0.0f)) {
        std::cerr << "--width, --height, --fps and --seconds-per-turn must be positive" << std::endl;
        return 1;
    }

    std::shared_ptr<klartraum::Font> unbounded, mono;
    try {
        unbounded = loadFont("Unbounded");
        mono = loadFont("JetBrainsMono");
    } catch (const std::exception& e) {
        std::cerr << e.what() << " (configure with KLARTRAUM_DOWNLOAD_FONTS=ON)" << std::endl;
        return 1;
    }

    const std::vector<klartraum::Gaussian3D> gaussians = klartraum::loadGaussiansSpz(spzFile, true);
    if (gaussians.empty()) {
        std::cerr << "no Gaussians loaded from " << spzFile << std::endl;
        return 1;
    }
    const glm::vec3 centre = sceneCentre(gaussians);

    klartraum::HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    const VkExtent2D extent{width, height};

    // The camera circles the vertical axis through the scene centre, as in
    // turntable_example; it starts at cameraPosition relative to the centre.
    const glm::vec3 cameraPosition(0.55f, 0.48f, 0.69f);
    const double startAzimuth = std::atan2(cameraPosition.z, cameraPosition.x);
    const double elevation = -std::asin(cameraPosition.y / glm::length(cameraPosition));
    klartraum::InterfaceCameraOrbit orbit(klartraum::InterfaceCameraOrbit::UpDirection::Y);
    orbit.initialize(vc);
    orbit.setPosition(-centre);
    orbit.setElevation(elevation);
    orbit.setDistance(timeline.cameraDistance(0));
    orbit.setAzimuth(startAzimuth);
    orbit.setProjectionAspectRatio(float(width) / float(height));
    klartraum::CameraMVP startCamera;
    orbit.update(startCamera);

    // Chapter 1: the title spans 70 % of the image width at the scene
    // centre's distance and is rasterized at twice the image resolution.
    const double distance = timeline.cameraDistance(0);
    const double focal = 0.5 * height / std::tan(0.5 * glm::radians(45.0));
    const float visibleWidth = float(width * distance / focal);
    const float size = 0.7f * visibleWidth / klartraum::measureLine({"klartraum", unbounded, 500.0f, 1.0f, 0.0f});
    klartraum::TitleLayout title;
    title.lines.push_back({"klartraum", unbounded, 500.0f, size, 0.0f});
    title.lines.push_back({"tech demo 000", mono, 400.0f, 0.4f * size, 0.05f});
    title.pixelsPerUnit = float(2.0 * focal / distance);
    const klartraum::TextMorphStart start = klartraum::prepareTextMorph(
        klartraum::sampleTextGaussians(klartraum::rasterizeTitle(title), klartraum::TextSplatParams{},
                                       uint32_t(gaussians.size())),
        gaussians, startCamera.view, centre);

    klartraum::MorphParams morph;
    morph.variant = klartraum::MorphVariant::SwirlScatter;
    morph.centre = centre;
    morph.axis = glm::vec3(0.0f, 1.0f, 0.0f);
    morph.scatterRadius = 0.25f;

    std::vector<CaptionBlock> captions;
    std::vector<klartraum::CaptionWindow> captionWindows;
    for (size_t c = 0; c < kChapters.size(); ++c) {
        captions.push_back(makeCaption(kChapters[c], unbounded, mono, height));
        captionWindows.push_back({int(c), kChapters[c].captionDelay, 0.5f, 0.5f});
    }

    auto target = std::make_shared<klartraum::OffscreenTarget>(vc, extent, 1u);
    auto cameraUBO = std::make_shared<klartraum::CameraUboType>();
    auto textData = std::make_shared<klartraum::GaussianDataStandard>(vc, start.gaussians);
    auto sceneData = std::make_shared<klartraum::GaussianDataStandard>(vc, gaussians);
    auto params = std::make_shared<klartraum::HostFloat>(vc, morph.values());
    const auto morphed =
        klartraum::createGaussianMorph(vc, textData->buffers(), sceneData->buffers(), klartraum::BufferRef{params},
                                       klartraum::uploadFloats(vc, start.delays, "MorphDelays"));

    // Chapter 7: the scene explodes. A second morph from the scene into
    // itself bursts the Gaussians away from the lantern; at progress 0 it
    // passes them through unchanged.
    klartraum::MorphParams explosion;
    explosion.variant = klartraum::MorphVariant::Explode;
    explosion.centre = centre;
    explosion.explodeShare = 0.95f;
    auto explosionValues = std::make_shared<klartraum::HostFloat>(vc, explosion.values());
    const auto exploded =
        klartraum::createGaussianMorph(vc, morphed.output, morphed.output, klartraum::BufferRef{explosionValues},
                                       klartraum::uploadFloats(vc, start.delays, "ExplosionDelays"));

    // Chapter 3: the morph's output is relit. The materials belong to the
    // scene's Gaussians, which the morph ends in; before chapter 3 the
    // relight value is 0, so the colours pass through unchanged.
    const klartraum::UprightObjectBounds lantern = klartraum::measureUprightObject(gaussians, 0.04f);
    auto lightingValues = std::make_shared<klartraum::HostValues<klartraum::LightingFrame>>(
        vc, std::vector<klartraum::LightingFrame>{klartraum::LightingFrame{}});
    const auto lit = klartraum::createGaussianLighting(vc, exploded.output, klartraum::bakeSplatMaterials(gaussians),
                                                       klartraum::BufferRef{lightingValues});

    // Chapters 4 and 5: the character. It is placed when chapter 4 starts:
    // half as tall as the lantern, joining a circle on the moss at the point
    // nearest the camera, from outside the picture at the side.
    const uint32_t walkStart = timeline.chapterStart(kCharacterChapter);
    const float sceneHeight = 0.5f * lantern.height;
    const float scale = sceneHeight / kCharacterHeight;
    const float pathRadius = lantern.footprintRadius + 0.5f * (lantern.groundRadius - lantern.footprintRadius);
    orbit.setAzimuth(startAzimuth + timeline.orbitAngle(walkStart));
    orbit.setDistance(timeline.cameraDistance(walkStart));
    klartraum::CameraMVP walkCamera;
    orbit.update(walkCamera);
    const glm::vec3 cameraAtWalk(walkCamera.cameraWorldPos);
    const glm::vec2 towards =
        glm::normalize(glm::vec2(cameraAtWalk.x - lantern.base.x, cameraAtWalk.z - lantern.base.z));
    const glm::vec3 entry = lantern.base + pathRadius * glm::vec3(towards.x, 0.0f, towards.y);
    const float halfWidth =
        float(std::tan(0.5 * glm::radians(45.0)) * double(width) / double(height)) * glm::distance(cameraAtWalk, entry);
    const float approach = 1.3f * halfWidth;
    const glm::mat4 characterToScene =
        klartraum::walkInToScene(lantern.base, pathRadius, cameraAtWalk, approach, sceneHeight, kCharacterHeight);
    std::cout << "character: walks " << approach / scale << " m in, then around a circle of " << pathRadius / scale
              << " m" << std::endl;

    auto controller = std::make_shared<klartraum::CharacterController>(vc, "data/character");
    klartraum::TitleLayout labelLayout;
    labelLayout.align = klartraum::TextAlign::Left;
    labelLayout.pixelsPerUnit = 1.0f;
    labelLayout.lines.push_back({"ControlNet input", mono, 400.0f, 17.0f * float(height) / 1080.0f, 0.0f});
    const klartraum::CoverageMask pictureLabel = klartraum::rasterizeTitle(labelLayout);
    klartraum::SkeletonGaussianParams skeletonParams;
    skeletonParams.toScene = characterToScene;
    skeletonParams.colourOpacity = glm::vec4(0.95f, 0.95f, 0.95f, 0.0f);
    skeletonParams.shape = glm::vec4(0.004f * lantern.height, 1.5f, 0.0f, 0.0f);
    auto skeletonValues = std::make_shared<klartraum::HostValues<klartraum::SkeletonGaussianParams>>(
        vc, std::vector<klartraum::SkeletonGaussianParams>{skeletonParams});
    const auto skeleton =
        klartraum::createSkeletonGaussians(vc, klartraum::BufferRef{controller->getJoints()},
                                           klartraum::characterSkeletonBones(), klartraum::BufferRef{skeletonValues});
    skeleton.element->addDependency(controller);
    // With --mesh-skeleton the skeleton Gaussians stay transparent and the mesh skeleton takes the parameters.
    auto meshSkeletonValues = std::make_shared<klartraum::HostValues<klartraum::SkeletonGaussianParams>>(
        vc, std::vector<klartraum::SkeletonGaussianParams>{skeletonParams});
    const auto merged = klartraum::createGaussianMerge(vc, lit.output, skeleton.output);

    const std::filesystem::path frameDir = outDir / "frames";
    std::filesystem::create_directories(frameDir);
    const uint32_t firstFrame = timeline.chapterStart(fromChapter - 1);
    const uint32_t endFrame = timeline.chapterStart(toChapter);
    const uint32_t morphEndFrame = uint32_t(std::ceil((timeline.titleSeconds + timeline.morphSeconds) * timeline.fps));
    std::cout << gaussians.size() << " Gaussians; rendering frames " << firstFrame << " to " << endFrame - 1 << " of "
              << timeline.frameCount() << std::endl;

    bool ok = true;
    {
        klartraum::GsplatConfig splatConfig;
        splatConfig.outputDepth = true;
        auto splatting = klartraum::createGaussianSplatting(vc, backend, target, cameraUBO, merged.output, splatConfig);
        auto direct = klartraum::createGaussianSplatting(vc, backend, target, cameraUBO, sceneData);

        // The pose map: the OpenPose skeleton alone on black, hidden where the
        // splatting's depth is in front of it. Its camera shares the scene
        // camera's view and projection, with the character's placement as model.
        auto skeletonCamera = std::make_shared<klartraum::CameraUboType>();
        auto barrier = std::make_shared<klartraum::BufferToGraphicsBarrier>();
        barrier->addBuffer(controller->getOpenPoseJoints());
        barrier->addBuffer(controller->getJoints());
        barrier->setInput(controller, 0);

        // With --mesh-skeleton: the skeleton as cylinders and spheres over the splatted scene, hidden where the
        // scene is in front of it.
        std::shared_ptr<klartraum::RenderPass> meshSkeletonPass;
        if (meshSkeleton) {
            auto meshRenderer = std::make_shared<klartraum::SkeletonMeshRenderer>(
                controller->getJoints(), klartraum::characterSkeletonBones(), meshSkeletonValues, extent);
            meshRenderer->setOcclusionDepth(klartraum::getGaussianSplattingDepth(splatting));
            // Splatting leaves the target ready for transfers; drawing over it needs GENERAL (see RenderPass).
            auto toGeneral = std::make_shared<klartraum::ImageViewSrcTransition>(
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
                VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
            toGeneral->setInput(splatting, 0, 0);
            meshSkeletonPass = std::make_shared<klartraum::RenderPass>(vc.getSwapChainImageFormat(), extent);
            meshSkeletonPass->setLoadExisting(true);
            meshSkeletonPass->setDepthBuffer(true);
            meshSkeletonPass->setInput(toGeneral, 0, 0);
            meshSkeletonPass->setInput(cameraUBO, 1);
            meshSkeletonPass->addDrawComponent(meshRenderer);
            meshSkeletonPass->addComputeDependency(barrier);
            meshSkeletonPass->addComputeDependency(meshSkeletonValues);
        }
        auto poseRenderer =
            std::make_shared<klartraum::OpenPoseSkeletonRenderer>(controller->getOpenPoseJoints(), 1, extent);
        poseRenderer->setOcclusionDepth(klartraum::getGaussianSplattingDepth(splatting));
        poseRenderer->setStickWidth(4.0f * float(height) / 1080.0f);
        poseRenderer->setJointRadius(5.0f * float(height) / 1080.0f);
        auto poseTarget = std::make_shared<klartraum::OffscreenTarget>(vc, extent, 1u);
        auto posePass = std::make_shared<klartraum::RenderPass>(vc.getSwapChainImageFormat(), extent);
        posePass->setInput(poseTarget, 0);
        posePass->setInput(skeletonCamera, 1);
        posePass->addDrawComponent(poseRenderer);
        posePass->addComputeDependency(barrier);
        posePass->addComputeDependency(splatting);
        if (meshSkeletonPass) {
            posePass->addComputeDependency(meshSkeletonPass);
        }

        // The character mask: the region of its body, cut by the scene in front.
        auto maskRenderer = std::make_shared<klartraum::CharacterMaskRenderer>(controller->getOpenPoseJoints(), extent);
        maskRenderer->setOcclusionDepth(klartraum::getGaussianSplattingDepth(splatting));
        auto maskTarget = std::make_shared<klartraum::OffscreenTarget>(vc, extent, 1u);
        auto maskPass = std::make_shared<klartraum::RenderPass>(vc.getSwapChainImageFormat(), extent);
        maskPass->setInput(maskTarget, 0);
        maskPass->setInput(skeletonCamera, 1);
        maskPass->addDrawComponent(maskRenderer);
        maskPass->addComputeDependency(barrier);
        maskPass->addComputeDependency(posePass);

        klartraum::ComputeGraph graph(vc, 1);
        graph.compileFrom(maskPass);
        klartraum::ComputeGraph directGraph(vc, 1);
        directGraph.compileFrom(direct);

        // Chapter 6 generates the person with SD1.5 + ControlNet, unless the
        // mask stands in for it.
        std::unique_ptr<PersonGenerator> generator;
        std::unique_ptr<GpuPersonGenerator> gpuGenerator;
        if (!placeholder && fromChapter <= kDiffusionChapter + 1 && toChapter >= kDiffusionChapter + 1) {
            const uint32_t latentSize = kGenerationSize / 8;
            const auto schedule =
                klartraum::LcmScheduler::load(controlDirectory(kGenerationSize) / "lcm_schedule_f32.bin");
            const auto noise = klartraum::NoiseField(schedule.size(), 4, latentSize, latentSize, seed)
                                   .window(0, 0, latentSize, latentSize);
            const auto embedding =
                klartraum::encodeSd15Prompt(vc, "data/onnx/sd15_denoiser_" + std::to_string(kGenerationSize), prompt);
            if (sdPath != "gpu") {
                generator = std::make_unique<PersonGenerator>(vc, kGenerationSize, extent, sceneData,
                                                              controller->getOpenPoseJoints(), characterToScene,
                                                              embedding, noise, backend);
            }
            if (sdPath != "cpu") {
                gpuGenerator = std::make_unique<GpuPersonGenerator>(
                    vc, kGenerationSize, extent, sceneData, controller->getOpenPoseJoints(), characterToScene,
                    embedding, noise, backend, cameraUBO, target, maskTarget, !dumpGraphs.empty());
            }
        }
        if (!dumpGraphs.empty() && !gpuGenerator) {
            std::cerr << "--dump-graphs needs the GPU path of chapter 6 (--from/--to including 6, no "
                         "--placeholder, --sd-path gpu or both)"
                      << std::endl;
            return 1;
        }
        // With --sd-path both: the GPU path's frames, and per frame the times of both paths and how much their
        // generated frames differ.
        const std::filesystem::path gpuFrameDir = outDir / "frames_gpu";
        std::ofstream comparison;
        if (generator && gpuGenerator) {
            std::filesystem::create_directories(gpuFrameDir);
            comparison.open(outDir / "sd_compare.csv");
            comparison << "frame,cpu_ms,gpu_ms,mean_abs_difference,max_abs_difference,psnr_db\n";
        }
        double cpuTotal = 0.0, gpuTotal = 0.0;
        uint32_t generatedFrames = 0;

        // Time of the per-frame graph submission, summed per chapter: milliseconds and count.
        std::vector<std::pair<double, uint32_t>> submitTimes(kChapters.size(), {0.0, 0u});
        klartraum::WalkInPathFollower follower(approach / scale, pathRadius / scale, kWalkSpeed, 0, kRunSpeed, 2.0f);
        // Runs the graph for frame f: camera, morph, lighting, character.
        auto submit = [&](klartraum::ComputeGraph& g, uint32_t f) {
            const int chapter = timeline.chapterAt(f);
            orbit.setAzimuth(startAzimuth + timeline.orbitAngle(f));
            orbit.setDistance(timeline.cameraDistance(f));
            orbit.update(cameraUBO->ubo);
            cameraUBO->update(0);
            skeletonCamera->ubo = cameraUBO->ubo;
            skeletonCamera->ubo.model = characterToScene;
            skeletonCamera->update(0);

            morph.progress = timeline.morphProgress(f);
            params->set(morph.values());
            explosion.explodeRadius = kExplosionRadius * lantern.height;
            // The progress stops where the burst comes to rest, before the
            // Explode variant would fly the Gaussians back into place.
            explosion.progress =
                chapter == kFinaleChapter
                    ? explosion.explodeShare * std::clamp(timeline.timeInChapter(f) / kExplosionSeconds, 0.0f, 1.0f)
                    : 0.0f;
            explosionValues->set(explosion.values());
            klartraum::LightingFrame lighting;
            if (chapter == kLightingChapter) {
                lighting = chapterLighting(timeline.timeInChapter(f), timeline.chapterSeconds[kLightingChapter],
                                           glm::vec3(cameraUBO->ubo.cameraWorldPos), lantern);
            }
            lightingValues->set(0, lighting);

            // The controller advances one step per run; the character starts
            // walking with chapter 4, from the controller's initial state.
            klartraum::CharacterControl control;
            control.velocity = glm::vec2(0.0f);
            if (f >= walkStart) {
                if (f == walkStart) {
                    controller->reset();
                    follower = klartraum::WalkInPathFollower(approach / scale, pathRadius / scale, kWalkSpeed, 0,
                                                             kRunSpeed, 2.0f);
                }
                const glm::vec4 hips = controller->readJoints()[0];
                control = follower.control({hips.x, hips.z});
            }
            controller->setControl(control);
            float skeletonOpacity = 0.0f;
            if (chapter == kCharacterChapter) {
                skeletonOpacity = 1.0f;
            } else if (chapter == kOpenPoseChapter) {
                skeletonOpacity = 1.0f - ramp(timeline.timeInChapter(f), 0.0f, 0.5f);
            }
            skeletonParams.colourOpacity.a = skeletonOpacity;
            (meshSkeleton ? meshSkeletonValues : skeletonValues)->set(0, skeletonParams);
            const auto start = std::chrono::steady_clock::now();
            g.submitAndWait(vc.getGraphicsQueue(), 0);
            if (&g == &graph) {
                submitTimes[chapter].first +=
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
                submitTimes[chapter].second += 1;
            }
        };

        std::vector<uint8_t> titleFrame;
        double titleDifference = 0.0, morphEndDifference = 0.0;
        // The character's walk depends on every step since chapter 4 began, so
        // those steps run even when their frames are not written.
        const uint32_t runFrom = firstFrame > walkStart ? walkStart : firstFrame;
        for (uint32_t f = runFrom; f < endFrame; ++f) {
            const bool written = f >= firstFrame && (f - firstFrame) % step == 0;
            if (!written && f < walkStart) {
                continue;
            }
            submit(graph, f);
            if (!written) {
                continue;
            }
            const int chapter = timeline.chapterAt(f);
            std::vector<uint8_t> frame = readImage(vc, target->getImage(0), extent);
            std::vector<uint8_t> gpuFrame; // with --sd-path both, the frame with the GPU path's person

            if (f < uint32_t(timeline.titleSeconds * timeline.fps)) {
                if (titleFrame.empty()) {
                    titleFrame = frame;
                } else {
                    titleDifference = std::max(titleDifference, meanAbsDifference(frame, titleFrame));
                }
            }
            if (f == morphEndFrame) {
                submit(directGraph, f);
                morphEndDifference = meanAbsDifference(frame, readImage(vc, target->getImage(0), extent));
            }

            if (chapter == kOpenPoseChapter) {
                const float t = timeline.timeInChapter(f);
                const auto pose = readImage(vc, poseTarget->getImage(0), extent, *poseTarget->getFinalLayoutOverride());
                overlayPose(frame, pose, ramp(t, 0.0f, 0.5f));
            } else if (chapter == kDiffusionChapter) {
                // The pose map gives way to the generated person (or, with
                // --placeholder, the character mask in its place); the pose
                // map stays, close up, as the ControlNet input.
                const float t = timeline.timeInChapter(f);
                const auto pose = readImage(vc, poseTarget->getImage(0), extent, *poseTarget->getFinalLayoutOverride());
                const auto mask = readImage(vc, maskTarget->getImage(0), extent, *maskTarget->getFinalLayoutOverride());
                if (generator || gpuGenerator) {
                    // The generated person blends in over the pose map, which stays close up as ControlNet's input.
                    auto finish = [&](std::vector<uint8_t> result, const GeneratedFrame& generated) {
                        const float shown = ramp(t, 0.0f, 0.5f);
                        for (size_t i = 0; i < result.size(); ++i) {
                            result[i] = uint8_t(float(result[i]) +
                                                (float(generated.full[i]) - float(result[i])) * shown + 0.5f);
                        }
                        overlayPose(result, pose, 1.0f - shown);
                        const VkExtent2D closeUp{kGenerationSize, kGenerationSize};
                        drawPictureInPicture(result, extent, generated.closeUpPose, closeUp,
                                             {0.0f, 0.0f, float(kGenerationSize)}, height * 3 / 10, pictureLabel,
                                             ramp(t, 1.0f, 1.5f));
                        return result;
                    };
                    using Clock = std::chrono::steady_clock;
                    GeneratedFrame cpuFrame, gpuResult;
                    double cpuMs = 0.0, gpuMs = 0.0;
                    if (generator) {
                        const auto start = Clock::now();
                        cpuFrame = generator->frame(frame, mask, controller->readOpenPoseJoints(), cameraUBO->ubo);
                        cpuMs = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
                    }
                    if (gpuGenerator) {
                        const auto start = Clock::now();
                        gpuResult = gpuGenerator->frame();
                        gpuMs = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
                    }
                    cpuTotal += cpuMs;
                    gpuTotal += gpuMs;
                    ++generatedFrames;
                    if (generator && gpuGenerator) {
                        double sum = 0.0, squares = 0.0;
                        int maximum = 0;
                        size_t count = 0;
                        for (size_t i = 0; i < frame.size(); i += 4) {
                            for (int c = 0; c < 3; ++c) {
                                const int d = std::abs(int(cpuFrame.full[i + c]) - int(gpuResult.full[i + c]));
                                sum += d;
                                squares += double(d) * d;
                                maximum = std::max(maximum, d);
                                ++count;
                            }
                        }
                        const double mse = squares / double(count);
                        const double psnr = mse > 0.0 ? 10.0 * std::log10(255.0 * 255.0 / mse) : 99.0;
                        comparison << f << "," << cpuMs << "," << gpuMs << "," << sum / double(count) << "," << maximum
                                   << "," << psnr << "\n";
                        comparison.flush();
                        gpuFrame = finish(frame, gpuResult);
                        std::cout << "generated frame " << f << ": cpu " << cpuMs << " ms, gpu " << gpuMs
                                  << " ms, PSNR " << psnr << " dB, max difference " << maximum << std::endl;
                    } else {
                        std::cout << "generated frame " << f << " in " << (generator ? cpuMs : gpuMs) << " ms"
                                  << std::endl;
                    }
                    frame = finish(frame, generator ? cpuFrame : gpuResult);
                } else {
                    overlayMask(frame, mask, 0.9f * ramp(t, 0.0f, 0.5f));
                    overlayPose(frame, pose, 1.0f - ramp(t, 0.0f, 0.5f));
                    const CropSquare crop =
                        characterCrop(controller->readOpenPoseJoints(),
                                      cameraUBO->ubo.proj * cameraUBO->ubo.view * characterToScene, extent, 1.4f);
                    drawPictureInPicture(frame, extent, pose, extent, crop, height * 3 / 10, pictureLabel,
                                         ramp(t, 1.0f, 1.5f));
                }
            }
            if (chapter == kFinaleChapter) {
                const float keep =
                    1.0f - ramp(timeline.timeInChapter(f), kFadeStart, timeline.chapterSeconds[kFinaleChapter]);
                for (size_t i = 0; i < frame.size(); i += 4) {
                    for (int c = 0; c < 3; ++c) {
                        frame[i + c] = uint8_t(float(frame[i + c]) * keep + 0.5f);
                    }
                }
            }
            drawCaption(frame, extent, captions[chapter], timeline.captionOpacity(captionWindows[chapter], f));
            char name[32];
            std::snprintf(name, sizeof(name), "frame_%04u.ppm", f);
            writePPM(frameDir / name, frame, extent);
            if (!gpuFrame.empty()) {
                drawCaption(gpuFrame, extent, captions[chapter], timeline.captionOpacity(captionWindows[chapter], f));
                writePPM(gpuFrameDir / name, gpuFrame, extent);
            }
            if (f == timeline.chapterStart(chapter)) {
                std::cout << "chapter " << chapter + 1 << " (" << kChapters[chapter].name << ") from frame " << f
                          << std::endl;
            }
        }
        if (endFrame > walkStart) {
            std::cout << "the character reached its circle: " << (follower.onCircle() ? "yes" : "no") << ", turns "
                      << follower.completedTurns() << std::endl;
        }
        if (!dumpGraphs.empty()) {
            gpuGenerator->exportGraphs(dumpGraphs);
            std::cout << "wrote the graphs of the SD1.5 chain, with their GPU times over " << generatedFrames
                      << " frames, to " << dumpGraphs.string() << std::endl;
        }
        if (generatedFrames > 0) {
            std::cout << "person generation: " << generatedFrames << " frames";
            if (generator) {
                std::cout << ", cpu path " << cpuTotal / generatedFrames << " ms per frame";
            }
            if (gpuGenerator) {
                std::cout << ", gpu path " << gpuTotal / generatedFrames << " ms per frame";
            }
            std::cout << std::endl;
        }
        for (size_t c = 0; c < kChapters.size(); ++c) {
            if (submitTimes[c].second > 0) {
                std::cout << "chapter " << c + 1 << " (" << kChapters[c].name << "): graph "
                          << submitTimes[c].first / submitTimes[c].second << " ms per frame over "
                          << submitTimes[c].second << " frames" << std::endl;
            }
        }
        std::cout << "title frames differ by " << titleDifference << ", the frame after the morph from the scene by "
                  << morphEndDifference << std::endl;
        // Progress 0 and 1 reproduce the text and the scene exactly.
        if (titleDifference > 0.0 || morphEndDifference > 0.0) {
            std::cerr << "the title does not hold still or the morph does not end in the scene" << std::endl;
            ok = false;
        }
    }
    target.reset();
    std::cout << "wrote frames " << firstFrame << " to " << endFrame - 1 << " to " << frameDir.string() << std::endl;
    return ok ? 0 : 1;
}
