// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Renders the opening shot of the promo video headlessly: the title
// "klartraum" / "tech demo 000" made of the scene's own Gaussians holds still,
// morphs into the lantern while the camera starts to circle it, and the
// lantern keeps turning. Writes one PPM per frame to
// <out-dir>/<variant>/frames/frame_XXXX.ppm, for one or all morph variants.
//
//   title_morph_example [--variant
//   straight|swirl|scatter|wave|explode|explode-swirl|shatter|swirl-scatter|swirl-wave|all] [--fps 30]
//                       [--title-seconds 2] [--morph-seconds 1.5] [--spin-seconds 5] [--turns 1]
//                       [--swirl-turns 1] [--scatter-radius 0.25] [--wave-spread 0.6]
//                       [--explode-radius 0.3] [--explode-share 0.35]
//                       [--file data/lantern.spz] [--no-flip-y]
//                       [--camera-position 0.55 0.48 0.69] [--distance 0.65]
//                       [--width 1920] [--height 1080] [--text-width 0.7]
//                       [--font Unbounded] [--weight 500] [--subtitle-font JetBrainsMono]
//                       [--title klartraum] [--subtitle "tech demo 000"]
//                       [--out-dir build/TestingOutput/promo_shot1]
//
// The text has exactly as many splats as the scene has Gaussians; each one
// flies to the Gaussian it is matched with (prepareTextMorph()), so nothing
// appears or disappears. The camera setup follows turntable_example.
//
// After each variant the program checks the shot: the title frames must be
// identical, and the last frame must equal the scene rendered directly from the
// same camera. It exits with an error otherwise.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/rendering/gaussianmorph.hpp"
#include "klartraum/computegraph/hostvalues.hpp"
#include "klartraum/rendering/gaussian_data_standard.hpp"
#include "klartraum/rendering/gaussian_morph.hpp"
#include "klartraum/rendering/gaussian_splatting_factory.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/interface_camera_orbit.hpp"
#include "klartraum/offscreen_target.hpp"
#include "klartraum/rendering/text_gaussians.hpp"

namespace {

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

struct Variant {
    const char* name;
    klartraum::MorphVariant variant;
};

const std::vector<Variant> kVariants = {
    {"straight", klartraum::MorphVariant::Straight},   {"swirl", klartraum::MorphVariant::Swirl},
    {"scatter", klartraum::MorphVariant::Scatter},     {"wave", klartraum::MorphVariant::Wave},
    {"explode", klartraum::MorphVariant::Explode},     {"explode-swirl", klartraum::MorphVariant::ExplodeSwirl},
    {"shatter", klartraum::MorphVariant::Shatter},     {"swirl-scatter", klartraum::MorphVariant::SwirlScatter},
    {"swirl-wave", klartraum::MorphVariant::SwirlWave}};

std::string fontPath(const std::string& name) {
    const bool isPath =
        name.size() > 4 && (name.substr(name.size() - 4) == ".ttf" || name.substr(name.size() - 4) == ".otf");
    return isPath ? name : "fonts/" + name + ".ttf";
}

} // namespace

int main(int argc, char** argv) {
    std::string variantName = "all";
    klartraum::MorphTimeline timeline;
    klartraum::MorphParams morph;
    morph.scatterRadius = 0.25f;
    std::string spzFile = "./data/lantern.spz";
    bool flipY = true;
    glm::vec3 cameraPosition(0.55f, 0.48f, 0.69f);
    double distance = 0.65;
    uint32_t width = 1920;
    uint32_t height = 1080;
    float textWidth = 0.7f;
    std::string fontName = "Unbounded";
    std::string subtitleFontName = "JetBrainsMono";
    float weight = 500.0f;
    std::string title = "klartraum";
    std::string subtitle = "tech demo 000";
    std::filesystem::path outDir = "build/TestingOutput/promo_shot1";

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const bool hasValue = i + 1 < argc;
        if (arg == "--variant" && hasValue) {
            variantName = argv[++i];
        } else if (arg == "--fps" && hasValue) {
            timeline.fps = std::stof(argv[++i]);
        } else if (arg == "--title-seconds" && hasValue) {
            timeline.titleSeconds = std::stof(argv[++i]);
        } else if (arg == "--morph-seconds" && hasValue) {
            timeline.morphSeconds = std::stof(argv[++i]);
        } else if (arg == "--spin-seconds" && hasValue) {
            timeline.spinSeconds = std::stof(argv[++i]);
        } else if (arg == "--turns" && hasValue) {
            timeline.turns = std::stof(argv[++i]);
        } else if (arg == "--swirl-turns" && hasValue) {
            morph.swirlTurns = std::stof(argv[++i]);
        } else if (arg == "--scatter-radius" && hasValue) {
            morph.scatterRadius = std::stof(argv[++i]);
        } else if (arg == "--wave-spread" && hasValue) {
            morph.waveSpread = std::stof(argv[++i]);
        } else if (arg == "--explode-radius" && hasValue) {
            morph.explodeRadius = std::stof(argv[++i]);
        } else if (arg == "--explode-share" && hasValue) {
            morph.explodeShare = std::stof(argv[++i]);
        } else if (arg == "--file" && hasValue) {
            spzFile = argv[++i];
        } else if (arg == "--no-flip-y") {
            flipY = false;
        } else if (arg == "--camera-position" && i + 3 < argc) {
            cameraPosition.x = std::stof(argv[++i]);
            cameraPosition.y = std::stof(argv[++i]);
            cameraPosition.z = std::stof(argv[++i]);
        } else if (arg == "--distance" && hasValue) {
            distance = std::stod(argv[++i]);
        } else if (arg == "--width" && hasValue) {
            width = uint32_t(std::stoul(argv[++i]));
        } else if (arg == "--height" && hasValue) {
            height = uint32_t(std::stoul(argv[++i]));
        } else if (arg == "--text-width" && hasValue) {
            textWidth = std::stof(argv[++i]);
        } else if (arg == "--font" && hasValue) {
            fontName = argv[++i];
        } else if (arg == "--weight" && hasValue) {
            weight = std::stof(argv[++i]);
        } else if (arg == "--subtitle-font" && hasValue) {
            subtitleFontName = argv[++i];
        } else if (arg == "--title" && hasValue) {
            title = argv[++i];
        } else if (arg == "--subtitle" && hasValue) {
            subtitle = argv[++i];
        } else if (arg == "--out-dir" && hasValue) {
            outDir = argv[++i];
        } else {
            std::cerr << "unknown or incomplete argument: " << arg << std::endl;
            return 1;
        }
    }

    std::vector<Variant> variants;
    for (const Variant& v : kVariants) {
        if (variantName == "all" || variantName == v.name) {
            variants.push_back(v);
        }
    }
    if (variants.empty()) {
        std::cerr
            << "--variant must be straight, swirl, scatter, wave, explode, explode-swirl, shatter, swirl-scatter, "
               "swirl-wave or all"
            << std::endl;
        return 1;
    }
    if (width == 0 || height == 0 || !(distance > 0.0) || !(timeline.fps > 0.0f) || timeline.frameCount() == 0 ||
        !(glm::length(cameraPosition) > 0.0f) || title.empty()) {
        std::cerr << "--width, --height, --distance, --fps and the shot length must be positive, "
                     "--camera-position must differ from the origin and the title must not be empty"
                  << std::endl;
        return 1;
    }

    const std::vector<klartraum::Gaussian3D> gaussians = klartraum::loadGaussiansSpz(spzFile, flipY);
    if (gaussians.empty()) {
        std::cerr << "no Gaussians loaded from " << spzFile << std::endl;
        return 1;
    }
    const glm::vec3 centre = sceneCentre(gaussians);
    morph.centre = centre;
    morph.axis = glm::vec3(0.0f, 1.0f, 0.0f); // the orbit camera circles the Y axis

    klartraum::HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    const VkExtent2D extent{width, height};
    const float aspect = float(width) / float(height);

    // The camera circles the vertical axis through the scene centre at a
    // fixed height, as in turntable_example.
    const double startAzimuth = std::atan2(cameraPosition.z, cameraPosition.x);
    const double elevation = -std::asin(cameraPosition.y / glm::length(cameraPosition));
    klartraum::InterfaceCameraOrbit orbit(klartraum::InterfaceCameraOrbit::UpDirection::Y);
    orbit.initialize(vc);
    orbit.setPosition(-centre);
    orbit.setElevation(elevation);
    orbit.setDistance(distance);
    orbit.setAzimuth(startAzimuth);
    orbit.setProjectionAspectRatio(aspect);
    klartraum::CameraMVP startCamera;
    orbit.update(startCamera);
    // The shatter flies towards where the camera is when the morph starts.
    morph.direction = glm::normalize(glm::vec3(startCamera.cameraWorldPos) - centre);

    // The title is set so the wordmark spans --text-width of the image at
    // the scene centre's distance, and rasterized at twice the image
    // resolution there.
    std::shared_ptr<klartraum::Font> font, subtitleFont;
    try {
        font = std::make_shared<klartraum::Font>(fontPath(fontName));
        subtitleFont = std::make_shared<klartraum::Font>(fontPath(subtitleFontName));
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    const double focal = 0.5 * height / std::tan(0.5 * glm::radians(45.0));
    const float visibleWidth = float(width * distance / focal);
    const float size = textWidth * visibleWidth / klartraum::measureLine({title, font, weight, 1.0f, 0.0f});
    klartraum::TitleLayout layout;
    layout.lines.push_back({title, font, weight, size, 0.0f});
    if (!subtitle.empty()) {
        layout.lines.push_back({subtitle, subtitleFont, 400.0f, 0.4f * size, 0.05f});
    }
    layout.pixelsPerUnit = float(2.0 * focal / distance);
    const klartraum::CoverageMask mask = klartraum::rasterizeTitle(layout);
    const klartraum::TextMorphStart start = klartraum::prepareTextMorph(
        klartraum::sampleTextGaussians(mask, klartraum::TextSplatParams{}, uint32_t(gaussians.size())), gaussians,
        startCamera.view, centre);
    std::cout << gaussians.size() << " Gaussians, " << timeline.frameCount() << " frames per variant" << std::endl;

    auto target = std::make_shared<klartraum::OffscreenTarget>(vc, extent, 1u);
    auto cameraUBO = std::make_shared<klartraum::CameraUboType>();
    auto textData = std::make_shared<klartraum::GaussianDataStandard>(vc, start.gaussians);
    auto sceneData = std::make_shared<klartraum::GaussianDataStandard>(vc, gaussians);
    auto params = std::make_shared<klartraum::HostFloat>(vc, morph.values());
    const auto morphed =
        klartraum::createGaussianMorph(vc, textData->buffers(), sceneData->buffers(), klartraum::BufferRef{params},
                                       klartraum::uploadFloats(vc, start.delays, "MorphDelays"));

    auto renderFrame = [&](klartraum::ComputeGraph& graph, double azimuth) {
        orbit.setAzimuth(azimuth);
        orbit.update(cameraUBO->ubo);
        cameraUBO->update(0);
        graph.submitAndWait(vc.getGraphicsQueue(), 0);
        return readImage(vc, target->getImage(0), extent);
    };

    bool ok = true;
    {
        auto splatting = klartraum::createGaussianSplatting(vc, klartraum::GsplatBackend::Compute, target, cameraUBO,
                                                            morphed.output);
        auto direct =
            klartraum::createGaussianSplatting(vc, klartraum::GsplatBackend::Compute, target, cameraUBO, sceneData);
        klartraum::ComputeGraph graph(vc, 1);
        graph.compileFrom(splatting);
        klartraum::ComputeGraph directGraph(vc, 1);
        directGraph.compileFrom(direct);

        for (const Variant& variant : variants) {
            const std::filesystem::path frameDir = outDir / variant.name / "frames";
            std::filesystem::create_directories(frameDir);
            morph.variant = variant.variant;
            std::vector<uint8_t> firstFrame, frame;
            double titleDifference = 0.0;
            const uint32_t last = timeline.frameCount() - 1;
            for (uint32_t f = 0; f <= last; ++f) {
                morph.progress = timeline.progressAt(f);
                params->set(morph.values());
                frame = renderFrame(graph, startAzimuth + timeline.angleAt(f));
                char name[32];
                std::snprintf(name, sizeof(name), "frame_%04u.ppm", f);
                writePPM(frameDir / name, frame, extent);
                if (f == 0) {
                    firstFrame = frame;
                } else if (f < timeline.titleFrames()) {
                    titleDifference = std::max(titleDifference, meanAbsDifference(frame, firstFrame));
                }
            }
            const double endDifference =
                meanAbsDifference(frame, renderFrame(directGraph, startAzimuth + timeline.angleAt(last)));
            std::cout << variant.name << ": wrote " << timeline.frameCount() << " frames to " << frameDir.string()
                      << "; title frames differ by " << titleDifference << ", last frame from the scene by "
                      << endDifference << std::endl;
            // Progress 0 and 1 reproduce the text and the scene exactly, so
            // both differences are zero unless the shot is broken.
            if (titleDifference > 0.0 || endDifference > 0.0) {
                std::cerr << variant.name << ": the title does not hold still or the morph does not end in the scene"
                          << std::endl;
                ok = false;
            }
        }
    }
    target.reset();
    return ok ? 0 : 1;
}
