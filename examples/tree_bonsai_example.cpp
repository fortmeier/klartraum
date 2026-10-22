// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Renders the bonsai forest (Kuth et al., Real-Time GPU Tree Generation, HPG
// 2025) headlessly from cameras around it, classically and as Gaussians, and
// writes one PPM image per view.
//
//   tree_bonsai_example [--renderer classical|gaussian|both] [--backend raster|compute]
//                       [--views 4] [--scale 0.21] [--relief 1.5] [--trees N] [--seed 7] [--width 1280] [--height 720]
//                       [--distance 4.2] [--elevation 0.32] [--azimuth 0.35] [--season 2]
//                       [--ground X,Z,METERS,HEADING] [--no-grass] [--profile] [--budget MS]
//                       [--video] [--days 4] [--seconds-per-day 10] [--fps 30] [--wind 0.5]
//                       [--turns 0.5] [--fly-in SECONDS] [--from 0] [--to N] [--moss-patches 250]
//                       [--out-dir build/TestingOutput/tree_bonsai]
//
// The trees are generated on the GPU every frame; per frame the CPU only sets
// the camera, the time, the season and the wind. Without --video, view i is
// seen from azimuth start + 2*pi*i/views at noon of the season, and images go
// to <out-dir>/frames/<renderer>_<view>.ppm. With --video, the frames of the
// days of BonsaiDays (one season per day, changing in the night, sun and moon
// moving with the season) go to <out-dir>/frames/<renderer>_<frame>.ppm while
// the camera turns --turns times around the pot; --from and --to select a
// range of frames. With --fly-in, the frames of a flight from the first view
// down to the --ground place go to <out-dir>/frames/<renderer>_<frame>.ppm, at
// noon of the season.

#include <algorithm>
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

float smootherstep(float x) {
    x = std::clamp(x, 0.0f, 1.0f);
    return x * x * x * (x * (6.0f * x - 15.0f) + 10.0f);
}

glm::vec3 slerpDirection(glm::vec3 a, glm::vec3 b, float t) {
    const float angle = std::acos(std::clamp(glm::dot(a, b), -1.0f, 1.0f));
    if (angle < 1e-4f) {
        return b;
    }
    return (std::sin((1.0f - t) * angle) * a + std::sin(t * angle) * b) / std::sin(angle);
}

struct FlyInPose {
    glm::vec3 eye;
    glm::vec3 lookAt;
    float nearPlane;
};

// The flight from `start` (looking at `focus`) down to `ground`, `above` units above the soil and looking along
// `heading` (radians from -z towards +x), at u in [0, 1]. The camera's distance to the place shrinks
// geometrically, so each halving of the distance takes the same time, while its direction from the place turns
// from the start's to behind the heading. The camera looks at the place and, in the last quarter, turns to look
// ahead; during the last quarter it pans slowly around. It stays above the soil, and the near plane follows its
// height above it.
FlyInPose flyInPose(const BonsaiScene& scene, glm::vec3 start, glm::vec3 focus, glm::vec2 ground, float above,
                    float heading, float u) {
    const float x = std::clamp(u / 0.8f, 0.0f, 1.0f);
    const float descent = x * x * (3.0f - 2.0f * x);
    const float pan = smootherstep((u - 0.75f) / 0.25f);
    const glm::vec3 place(ground.x, scene.soilHeight(ground.x, ground.y) + above, ground.y);
    const float h = heading + glm::radians(35.0f) * pan;
    const glm::vec3 forward(std::sin(h), 0.0f, -std::cos(h));
    const glm::vec3 behind = glm::normalize(-forward + glm::vec3(0.0f, 0.25f, 0.0f));
    const float startDistance = glm::length(start - place);
    const float endDistance = 2.0f * above;
    const float d = startDistance * std::pow(endDistance / startDistance, descent);
    const glm::vec3 dir = slerpDirection(glm::normalize(start - place), behind, descent);
    glm::vec3 eye = place + d * dir;
    const float soil = scene.soilHeight(eye.x, eye.z);
    eye.y = std::max(eye.y, soil + above);

    const float ahead = smootherstep((descent - 0.75f) / 0.25f);
    const glm::vec3 toPlace = glm::normalize(glm::mix(focus, place, smootherstep(descent / 0.3f)) - eye);
    const glm::vec3 lookAhead = glm::normalize(forward + glm::vec3(0.0f, -0.08f, 0.0f));
    const glm::vec3 look = slerpDirection(toPlace, lookAhead, ahead);
    return {eye, eye + look, std::clamp(0.5f * (eye.y - soil), 1e-5f, 0.05f)};
}

// `nearPlane` follows how close the camera is to the ground: from outside the pot 0.05 units, standing in a large
// landscape a fraction of a metre.
void setCamera(CameraMVP& ubo, glm::vec3 eye, glm::vec3 target, float aspect, float nearPlane = 0.05f) {
    ubo.model = glm::mat4(1.0f);
    ubo.view = glm::lookAt(eye, target, glm::vec3(0.0f, 1.0f, 0.0f));
    ubo.proj = glm::perspective(glm::radians(40.0f), aspect, nearPlane, 50.0f);
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
    float season = 2.0f;
    bool video = false;
    bool profile = false;
    bool grass = true;
    float budgetMs = 0.0f;
    bool ground = false;
    glm::vec4 groundView(0.0f); // x, z in units, metres above the ground, heading in degrees (0 towards -z)
    BonsaiDays days;
    float fps = 30.0f;
    float windStrength = 0.5f;
    float turns = 0.5f;
    float flyInSeconds = 0.0f;
    uint32_t mossPatches = BonsaiRenderConfig{}.mossPatches;
    uint32_t fromFrame = 0;
    uint32_t toFrame = UINT32_MAX;
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
        } else if (arg == "--scale" && hasValue) {
            sceneConfig.scale.s = std::stof(argv[++i]);
        } else if (arg == "--relief" && hasValue) {
            sceneConfig.scale.reliefExaggeration = std::stof(argv[++i]);
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
        } else if (arg == "--season" && hasValue) {
            season = std::stof(argv[++i]);
        } else if (arg == "--ground" && hasValue) {
            ground =
                std::sscanf(argv[++i], "%f,%f,%f,%f", &groundView.x, &groundView.y, &groundView.z, &groundView.w) == 4;
        } else if (arg == "--budget" && hasValue) {
            budgetMs = std::stof(argv[++i]);
        } else if (arg == "--no-grass") {
            grass = false;
        } else if (arg == "--profile") {
            profile = true;
        } else if (arg == "--video") {
            video = true;
        } else if (arg == "--days" && hasValue) {
            days.days = uint32_t(std::stoul(argv[++i]));
        } else if (arg == "--seconds-per-day" && hasValue) {
            days.secondsPerDay = std::stof(argv[++i]);
        } else if (arg == "--fps" && hasValue) {
            fps = std::stof(argv[++i]);
        } else if (arg == "--wind" && hasValue) {
            windStrength = std::stof(argv[++i]);
        } else if (arg == "--turns" && hasValue) {
            turns = std::stof(argv[++i]);
        } else if (arg == "--fly-in" && hasValue) {
            flyInSeconds = std::stof(argv[++i]);
        } else if (arg == "--moss-patches" && hasValue) {
            mossPatches = uint32_t(std::stoul(argv[++i]));
        } else if (arg == "--from" && hasValue) {
            fromFrame = uint32_t(std::stoul(argv[++i]));
        } else if (arg == "--to" && hasValue) {
            toFrame = uint32_t(std::stoul(argv[++i]));
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
    if (flyInSeconds > 0.0f && !ground) {
        std::cerr << "--fly-in needs --ground for the place it flies to" << std::endl;
        return 1;
    }

    HeadlessFrontend frontend;
    VulkanContext& vc = frontend.getKlartraumEngine().getVulkanContext();
    const BonsaiScene scene = makeBonsaiScene(sceneConfig);
    std::filesystem::create_directories(outDir / "frames");
    std::cout << "scale " << sceneConfig.scale.s << ": the pot holds " << sceneConfig.scale.regionMeters() << " m, "
              << scene.trees.size() << " trees, lantern " << sceneConfig.scale.lanternMeters() << " m" << std::endl;

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
        config.mossPatches = mossPatches;
        config.grass = grass;

        auto target = std::make_shared<OffscreenTarget>(vc, extent, 1);
        BonsaiRenderer bonsai(vc, scene, target, config);
        ComputeGraph graph(vc, 1);
        if (profile) {
            graph.enableProfiling();
        }
        graph.compileFrom(bonsai.getOutput());
        bonsai.initialize(1);
        bonsai.setSeason(season);

        if (flyInSeconds > 0.0f) {
            const uint32_t frames = uint32_t(std::lround(flyInSeconds * fps));
            const glm::vec3 start =
                scene.focus + distance * glm::vec3(std::sin(azimuth) * std::cos(elevation), std::sin(elevation),
                                                   std::cos(azimuth) * std::cos(elevation));
            const float above = groundView.z / scene.config.scale.metersPerUnit();
            for (uint32_t frame = fromFrame; frame < std::min(frames, toFrame); ++frame) {
                const float seconds = float(frame) / fps;
                const FlyInPose pose =
                    flyInPose(scene, start, scene.focus, glm::vec2(groundView.x, groundView.y), above,
                              glm::radians(groundView.w), float(frame) / float(std::max(frames - 1, 1u)));
                setCamera(bonsai.getCamera()->ubo, pose.eye, pose.lookAt, float(extent.width) / float(extent.height),
                          pose.nearPlane);
                bonsai.getCamera()->update(0);
                bonsai.setView(pose.eye);
                bonsai.setTime(seconds, 0.5f, season,
                               glm::vec4(glm::normalize(glm::vec3(1.0f, 0.0f, 0.35f)), windStrength));
                // Tiles of the placement come a few per frame; a second frame lets them keep up with the flight.
                graph.submitAndWait(vc.getGraphicsQueue(), 0);
                bonsai.setView(pose.eye);
                graph.submitAndWait(vc.getGraphicsQueue(), 0);
                char name[64];
                std::snprintf(name, sizeof(name), "%s_%04u.ppm", modeName, frame);
                writePpm(outDir / "frames" / name, readImage(vc, target->getImage(0), extent), extent);
                if (frame % 30 == 0) {
                    std::cout << modeName << " frame " << frame << " / " << frames << std::endl;
                }
            }
            continue;
        }
        if (video) {
            const uint32_t frames = uint32_t(std::lround(days.duration() * fps));
            TreeDetailBudget budget(budgetMs > 0.0f ? budgetMs : 1e9f);
            const float aspect = float(extent.width) / float(extent.height);
            for (uint32_t frame = fromFrame; frame < std::min(frames, toFrame); ++frame) {
                const float seconds = float(frame) / fps;
                const float angle = azimuth + 2.0f * 3.14159265f * turns * seconds / days.duration();
                const glm::vec3 eye =
                    scene.focus + distance * glm::vec3(std::sin(angle) * std::cos(elevation), std::sin(elevation),
                                                       std::cos(angle) * std::cos(elevation));
                setCamera(bonsai.getCamera()->ubo, eye, scene.focus, aspect);
                bonsai.getCamera()->update(0);
                bonsai.setView(eye);
                bonsai.setTime(seconds, days.timeOfDayAt(seconds), days.seasonAt(seconds),
                               glm::vec4(glm::normalize(glm::vec3(1.0f, 0.0f, 0.35f)), windStrength));
                const auto start = std::chrono::steady_clock::now();
                graph.submitAndWait(vc.getGraphicsQueue(), 0);
                if (budgetMs > 0.0f) {
                    // The next frame's detail from this frame's time.
                    bonsai.setDetail(budget.update(float(
                        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count())));
                }
                char name[64];
                std::snprintf(name, sizeof(name), "%s_%04u.ppm", modeName, frame);
                writePpm(outDir / "frames" / name, readImage(vc, target->getImage(0), extent), extent);
                if (frame % 30 == 0) {
                    std::cout << modeName << " frame " << frame << " / " << frames << std::endl;
                }
            }
            continue;
        }

        for (uint32_t view = 0; view < views; ++view) {
            const float angle = azimuth + 2.0f * 3.14159265f * float(view) / float(views);
            glm::vec3 eye =
                scene.focus + distance * glm::vec3(std::sin(angle) * std::cos(elevation), std::sin(elevation),
                                                   std::cos(angle) * std::cos(elevation));
            glm::vec3 lookAt = scene.focus;
            float nearPlane = 0.05f;
            if (ground) {
                // Standing in the landscape, looking a little down along the heading (turned by the view).
                const float above = groundView.z / scene.config.scale.metersPerUnit();
                eye = glm::vec3(groundView.x, scene.soilHeight(groundView.x, groundView.y) + above, groundView.y);
                const float heading = glm::radians(groundView.w) + 2.0f * 3.14159265f * float(view) / float(views);
                lookAt = eye + glm::vec3(std::sin(heading), -0.08f, -std::cos(heading));
                nearPlane = std::clamp(0.5f * above, 1e-5f, 0.05f);
            }
            setCamera(bonsai.getCamera()->ubo, eye, lookAt, float(extent.width) / float(extent.height), nearPlane);
            bonsai.getCamera()->update(0);
            bonsai.setView(eye);
            // A still waits for all tiles of the placement in range, which come a few per frame, and for the
            // canopy's band to settle.
            for (int settle = 0; settle < 32 && !bonsai.isSettled(); ++settle) {
                graph.submitAndWait(vc.getGraphicsQueue(), 0);
                bonsai.setView(eye);
            }
            const auto start = std::chrono::steady_clock::now();
            graph.submitAndWait(vc.getGraphicsQueue(), 0);
            const double ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            char name[64];
            std::snprintf(name, sizeof(name), "%s_%03u.ppm", modeName, view);
            writePpm(outDir / "frames" / name, readImage(vc, target->getImage(0), extent), extent);
            std::cout << modeName << " view " << view << ": " << ms << " ms" << std::endl;
            if (profile) {
                // The last frame's GPU times, longest first.
                graph.resetProfiling();
                graph.submitAndWait(vc.getGraphicsQueue(), 0);
                auto results = graph.getProfilingResults();
                std::sort(results.begin(), results.end(),
                          [](const auto& a, const auto& b) { return a.second > b.second; });
                std::cout << "  GPU span " << graph.getGpuSpanMs() << " ms" << std::endl;
                for (size_t k = 0; k < results.size() && k < 12; ++k) {
                    std::cout << "  " << results[k].first << ": " << results[k].second << " ms" << std::endl;
                }
            }
        }
    }
    return 0;
}
