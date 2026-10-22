// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Measures how much a single Gaussian tree pops (poppingResidual()) and how far it lies from the same tree without
// levels of detail (fidelityError()) while the camera moves away from it and back, as the Gaussian BonsaiRenderer
// draws it today: generated every frame (TreeGaussians) while it is among the near trees, beyond that as its
// library copy (TreeMossLod) with the copy's levels of detail by size on screen:
//
//   tree_lod_metrics_example [--species broadleaf|conifer] [--yaw 0.3] [--width 1280] [--height 720] [--fov 50]
//                            [--start 5] [--far 300] [--frames 800] [--near-radius M] [--wind 0.5]
//                            [--out-dir build/TestingOutput/tree_lod_metrics]
//
// Four runs over the same flight: a static tree (no wind) and a dynamic one (wind of --wind, as the bonsai explorer
// blows it), each with the levels of detail and without them (generated every frame at every distance without
// any reduction by size), the reference of the fidelity. The camera, level with the tree's middle, moves from --start
// to
// --far metres from the trunk and back, the logarithm of the distance following a cosine, so the motion is smooth
// everywhere and slow at both ends. The tree is generated while the camera is nearer than --near-radius; by default
// that is the near radius at which the renderer hands its trees over to the library at the bonsai explorer's start
// in the forest (0.8 times the distance of the scene's nearTrees-th nearest tree there). The copy is never squeezed
// by its pool here (one tree).
//
// Writes <out-dir>/metrics.csv (per frame: distance, what draws the tree, the copy's level and crossfade, the
// crown's diameter in pixels, the four runs' popping residuals, the fidelity errors of the static and the dynamic
// run and the four runs' GPU times) and <out-dir>/frames/frame_NNNN.ppm (with levels of detail above, without below;
// static left, dynamic right; each at 3/4 of the measured size), and prints a summary.
// scripts/tree/make_lod_metrics_video.sh labels the frames and encodes the video.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <glm/gtc/matrix_transform.hpp>

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/gaussian_splatting_factory.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/offscreen_target.hpp"
#include "klartraum/lod_metrics.hpp"
#include "klartraum/tree/bonsai_flight.hpp"
#include "klartraum/tree/bonsai_scene.hpp"
#include "klartraum/tree/tree_gaussians.hpp"
#include "klartraum/tree/tree_generator.hpp"
#include "klartraum/tree/tree_library.hpp"
#include "klartraum/tree/tree_moss_lod.hpp"
#include "klartraum/tree/tree_placement.hpp"

using namespace klartraum;

namespace {

constexpr float kSeason = 2.0f;

// Reads an OffscreenTarget image back as tightly packed BGRA bytes; the splatting leaves it in TRANSFER_SRC_OPTIMAL.
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

// Bilinear samples of a BGRA image at the centres of a smaller image's pixels, as RGB.
void resampleInto(const std::vector<uint8_t>& bgra, VkExtent2D from, uint8_t* rgb, uint32_t rowWidth, uint32_t x0,
                  VkExtent2D to) {
    const float sx = float(from.width) / float(to.width), sy = float(from.height) / float(to.height);
    for (uint32_t y = 0; y < to.height; ++y) {
        const float fy = std::clamp((float(y) + 0.5f) * sy - 0.5f, 0.0f, float(from.height - 1));
        const uint32_t y0 = uint32_t(fy), y1 = std::min(y0 + 1, from.height - 1);
        const float wy = fy - float(y0);
        for (uint32_t x = 0; x < to.width; ++x) {
            const float fx = std::clamp((float(x) + 0.5f) * sx - 0.5f, 0.0f, float(from.width - 1));
            const uint32_t xa = uint32_t(fx), xb = std::min(xa + 1, from.width - 1);
            const float wx = fx - float(xa);
            for (int c = 0; c < 3; ++c) {
                auto at = [&](uint32_t px, uint32_t py) {
                    return float(bgra[(size_t(py) * from.width + px) * 4 + 2 - c]);
                };
                const float v = (1.0f - wy) * ((1.0f - wx) * at(xa, y0) + wx * at(xb, y0)) +
                                wy * ((1.0f - wx) * at(xa, y1) + wx * at(xb, y1));
                rgb[(size_t(y) * rowWidth + x0 + x) * 3 + c] = uint8_t(std::lround(std::clamp(v, 0.0f, 255.0f)));
            }
        }
    }
}

// One tree over black, drawn by the raster splatting into its own target: either generated every frame as the
// near trees are (TreeGaussians) or as its library copy (TreeMossLod), lit the same way, in season kSeason.
// With `lodFree` the generated tree has no level of detail: no stems culled or drawn round by size
// (TreeRenderSettings::viewport.w, gaussianLod) and its leaves in full (leafLod).
struct TreeView {
    TreeView(VulkanContext& vc, const BonsaiScene& scene, const TreeSite& site, bool copy, VkExtent2D extent,
             std::shared_ptr<CameraUboType> camera, const std::vector<GaussianLods>& library, uint32_t variant,
             const TreeGaussianForestConfig& forest, bool lodFree = false, glm::vec4 leafLod = glm::vec4(-1.0f),
             float stemRound = -1.0f, float stemFade = -1.0f)
        : vc(vc),
          extent(extent) {
        settings = std::make_shared<HostValues<TreeRenderSettings>>(
            vc, std::vector<TreeRenderSettings>{defaultTreeRenderSettings(extent)});
        TreeRenderSettings values = settings->values().at(0);
        values.season = glm::vec4(kSeason, tree::treeSnowCover(kSeason), 0.0f, 0.0f);
        if (leafLod.x >= 0.0f) {
            values.leafLod = leafLod;
        }
        if (stemRound >= 0.0f) {
            values.gaussianLod.x = stemRound;
        }
        if (stemFade >= 0.0f) {
            values.gaussianLod.y = stemFade;
        }
        if (lodFree) {
            values.viewport.w = 0.0f;
            values.gaussianLod = glm::vec4(0.0f);
            values.leafLod = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
        }
        settings->set(0, values);
        TreeFrame frame = defaultTreeFrame();
        frame.timeSeason.y = kSeason;
        TreeGaussianSources sources;
        const uint32_t type = site.kind.x == TREE_SITE_CONIFER ? 1u : 0u;
        if (copy) {
            // A generator of a trunk only, far away; the copy goes into the extra slots.
            TreeTypeParams p = tree_presets::quakingAspen();
            p.ratios.x = 1.0f;
            p.leafCount.x = 0.0f;
            const std::vector<TreeTypeParams> types = {p};
            TreeInstance far{};
            far.posScale = glm::vec4(1e4f, 0.0f, 0.0f, 0.01f);
            far.rot = treeUpRotation(glm::vec3(0.0f, 1.0f, 0.0f));
            far.ids = glm::uvec4(0u, 1u, 0u, 0u);
            const std::vector<TreeInstance> trees = {far};
            generator =
                std::make_shared<TreeGenerator>(vc, types, trees, TreeCapacity::fromReference(types, trees, frame));
            sources.trees = false;
            // The splatting runs over every slot, used or not: room for level 0 and level 1 while crossfading.
            sources.extraSlots = 4096;
        } else {
            const std::vector<TreeInstance> trees = {
                treeInstanceFor(site, scene.types, scene.config.seed, forest.variants)};
            generator = std::make_shared<TreeGenerator>(vc, scene.types, trees,
                                                        TreeCapacity::fromReference(scene.types, trees, frame), frame);
        }
        gaussians = std::make_unique<TreeGaussians>(vc, *generator, camera, settings, std::move(sources));
        GaussianSoABuffers splats = gaussians->getGaussians();
        if (copy) {
            // As tree_library_instances.comp writes the copy: turned and scaled like the generated tree, fully
            // faded in, in the sun.
            TreeLodInstance inst{};
            inst.posYaw = glm::vec4(glm::vec3(site.posHeight), site.kind.y);
            inst.plane = glm::vec4(0.0f, 0.0f, site.posHeight.w / scene.types[type].shapeBase.z, 0.0f);
            inst.ids = glm::uvec4(variant, glm::floatBitsToUint(0.0f), glm::floatBitsToUint(0.0f), 1u);
            lod = std::make_unique<TreeMossLod>(vc, *gaussians, library, std::vector<TreeLodInstance>{inst}, camera,
                                                settings, BufferRef{}, forest.boundaries);
            splats = lod->getGaussians();
            for (const Gaussian3D& g : library[variant].levels[0]) {
                crownRadius = std::max(crownRadius, glm::length(glm::vec2(g.position[0], g.position[2])));
            }
            crownRadius *= inst.plane.z;
        }
        target = std::make_shared<OffscreenTarget>(vc, extent, 1);
        splatting = createGaussianSplatting(vc, GsplatBackend::Raster, target, camera, splats);
        graph = std::make_unique<ComputeGraph>(vc, 1);
        graph->enableProfiling();
        graph->compileFrom(splatting);
        gaussians->initializeBuffers(1);
    }

    std::vector<uint8_t> render(float seconds, glm::vec4 wind) {
        TreeFrame frame = generator->getFrame().values().at(0);
        frame.timeSeason = glm::vec4(seconds, kSeason, 0.0f, 0.0f);
        frame.wind = wind;
        generator->getFrame().set(0, frame);
        graph->resetProfiling();
        graph->submitAndWait(vc.getGraphicsQueue(), 0);
        gpuMs = float(graph->getGpuSpanMs());
        return readImage(vc, target->getImage(0), extent);
    }

    VulkanContext& vc;
    VkExtent2D extent;
    float crownRadius = 0.0f; ///< of the copy, as TreeMossLod measures it for its levels
    float gpuMs = 0.0f;       ///< GPU time of the last render(), from generation to splatting
    std::shared_ptr<HostValues<TreeRenderSettings>> settings;
    std::shared_ptr<TreeGenerator> generator;
    std::unique_ptr<TreeGaussians> gaussians;
    std::unique_ptr<TreeMossLod> lod;
    std::shared_ptr<OffscreenTarget> target;
    std::shared_ptr<ComputeGraphElement> splatting;
    std::unique_ptr<ComputeGraph> graph;
};

// The popping of a run: its largest residual, and those where what draws the tree changes against the run's
// median, the frames around it and the same frames without levels of detail (`reference`, the residual the motion
// alone leaves).
void printPopping(const char* name, const std::vector<float>& residuals, const std::vector<float>& reference,
                  const std::vector<float>& distances, const std::vector<std::string>& drawnBy) {
    // The first and last frames have no neighbour on one side.
    const std::vector<float> inner(residuals.begin() + 1, residuals.end() - 1);
    const SeriesSummary s = summarizeSeries(inner);
    const size_t worst = s.worst + 1;
    const SeriesSummary free = summarizeSeries(std::vector<float>(reference.begin() + 1, reference.end() - 1));
    std::printf("%s popping: max %.5f at frame %zu (%.1f m, %s), median %.5f, mean %.5f, max/median %.1f; without "
                "levels of detail max %.5f, median %.5f\n",
                name, s.max, worst, distances[worst], drawnBy[worst].c_str(), s.median, s.mean, s.spike, free.max,
                free.median);
    // The frames where what draws the tree changes, and their residuals (the larger of the two frames around it),
    // against the run's median and against the frames 3 to 12 away on both sides (the motion there).
    for (size_t k = 1; k + 2 < residuals.size(); ++k) {
        if (drawnBy[k] != drawnBy[k + 1]) {
            const float r = std::max(residuals[k], residuals[k + 1]);
            std::vector<float> around;
            for (size_t j = k >= 12 ? k - 12 : 1; j + 2 <= k; ++j) {
                around.push_back(residuals[j]);
            }
            for (size_t j = k + 4; j < std::min(k + 14, residuals.size() - 1); ++j) {
                around.push_back(residuals[j]);
            }
            const float local = summarizeSeries(around).median;
            const float motion = std::max(reference[k], reference[k + 1]);
            std::printf("  %-12s -> %-12s at %6.1f m: %.5f (%5.1f x median, %5.1f x around, %5.1f x without levels "
                        "of detail)\n",
                        drawnBy[k].c_str(), drawnBy[k + 1].c_str(), 0.5f * (distances[k] + distances[k + 1]), r,
                        s.median > 0.0f ? r / s.median : 0.0f, local > 0.0f ? r / local : 0.0f,
                        motion > 0.0f ? r / motion : 0.0f);
        }
    }
}

// Per-frame values of a run (fidelity errors, GPU times): their mean and largest over all frames and over the
// frames drawn by each representation.
void printByRepresentation(const char* name, const std::vector<float>& values, const std::vector<float>& distances,
                           const std::vector<std::string>& drawnBy) {
    const std::vector<float>& errors = values;
    const SeriesSummary all = summarizeSeries(errors);
    std::printf("%s: mean %.4f, max %.4f at %.1f m (%s)\n", name, all.mean, all.max, distances[all.worst],
                drawnBy[all.worst].c_str());
    std::vector<std::string> order;
    for (const std::string& by : drawnBy) {
        if (std::find(order.begin(), order.end(), by) == order.end()) {
            order.push_back(by);
        }
    }
    for (const std::string& by : order) {
        std::vector<float> e;
        float nearest = 1e30f, farthest = 0.0f;
        for (size_t k = 0; k < errors.size(); ++k) {
            if (drawnBy[k] == by) {
                e.push_back(errors[k]);
                nearest = std::min(nearest, distances[k]);
                farthest = std::max(farthest, distances[k]);
            }
        }
        const SeriesSummary s = summarizeSeries(e);
        std::printf("  %-12s %6.1f - %6.1f m: mean %.4f, max %.4f\n", by.c_str(), nearest, farthest, s.mean, s.max);
    }
}

} // namespace

int main(int argc, char** argv) {
    std::string species = "broadleaf";
    float yaw = 0.3f;
    uint32_t width = 1280, height = 720;
    float fov = 50.0f;
    float startMeters = 5.0f, farMeters = 300.0f;
    int frames = 800;
    float nearRadiusMeters = 0.0f; // 0: from the forest at the explorer's start
    float windStrength = 0.5f;
    glm::vec4 leafLod(-1.0f); // < 0: the renderer's default
    float stemRound = -1.0f;
    float stemFade = -1.0f;
    std::filesystem::path outDir = "build/TestingOutput/tree_lod_metrics";
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const bool hasValue = i + 1 < argc;
        if (arg == "--species" && hasValue) {
            species = argv[++i];
        } else if (arg == "--yaw" && hasValue) {
            yaw = std::stof(argv[++i]);
        } else if (arg == "--width" && hasValue) {
            width = uint32_t(std::stoul(argv[++i]));
        } else if (arg == "--height" && hasValue) {
            height = uint32_t(std::stoul(argv[++i]));
        } else if (arg == "--fov" && hasValue) {
            fov = std::stof(argv[++i]);
        } else if (arg == "--start" && hasValue) {
            startMeters = std::stof(argv[++i]);
        } else if (arg == "--far" && hasValue) {
            farMeters = std::stof(argv[++i]);
        } else if (arg == "--frames" && hasValue) {
            frames = std::stoi(argv[++i]);
        } else if (arg == "--near-radius" && hasValue) {
            nearRadiusMeters = std::stof(argv[++i]);
        } else if (arg == "--leaf-lod" && hasValue) {
            if (std::sscanf(argv[++i], "%f,%f,%f,%f", &leafLod.x, &leafLod.y, &leafLod.z, &leafLod.w) != 4) {
                std::cerr << "--leaf-lod takes x,y,z,w" << std::endl;
                return 1;
            }
        } else if (arg == "--stem-round" && hasValue) {
            stemRound = std::stof(argv[++i]);
        } else if (arg == "--stem-fade" && hasValue) {
            stemFade = std::stof(argv[++i]);
        } else if (arg == "--wind" && hasValue) {
            windStrength = std::stof(argv[++i]);
        } else if (arg == "--out-dir" && hasValue) {
            outDir = argv[++i];
        } else {
            std::cerr << "unknown option " << arg << std::endl;
            return 1;
        }
    }
    if (species != "broadleaf" && species != "conifer") {
        std::cerr << "--species must be broadleaf or conifer" << std::endl;
        return 1;
    }
    if (frames < 4 || startMeters <= 0.0f || farMeters <= startMeters) {
        std::cerr << "need --frames >= 4 and 0 < --start < --far" << std::endl;
        return 1;
    }

    // The landscape at scale 1 for its tree types and sizes.
    BonsaiSceneConfig sceneConfig;
    sceneConfig.scale.s = 1.0f;
    sceneConfig.seed = 7;
    const BonsaiScene scene = makeBonsaiScene(sceneConfig);
    const float m = 1.0f / scene.config.scale.metersPerUnit(); // units per metre
    if (nearRadiusMeters <= 0.0f) {
        std::cout << "finding the explorer's place in the forest ..." << std::endl;
        const std::vector<ForestApproach> approaches = findForestApproach(scene);
        if (approaches.empty()) {
            std::cerr << "no forest found; give --near-radius" << std::endl;
            return 1;
        }
        const glm::vec2 place = approaches.front().end;
        const glm::vec3 eye(place.x, scene.soilHeight(place.x, place.y) + 1.7f * m, place.y);
        const std::vector<TreeSite> sites =
            nearestTreeSites(scene.terrain, scene.placement, eye, scene.config.nearTrees);
        // As BonsaiRenderer::setView() settles there: 0.8 times the distance of the nearTrees-th nearest tree.
        nearRadiusMeters = 0.8f * glm::length(glm::vec3(sites.back().posHeight) - eye) / m;
    }

    TreeSite site{};
    site.posHeight = glm::vec4(0.0f, 0.0f, 0.0f, scene.config.scale.treeHeightUnits());
    site.kind = glm::vec4(species == "conifer" ? TREE_SITE_CONIFER : TREE_SITE_BROADLEAF, yaw, 1.0f, 1.0f);
    const uint32_t type = species == "conifer" ? 1u : 0u;
    const float treeHeight = site.posHeight.w;

    // The library copies of the tree's type in this season, one per variant; the site's variant is the one the
    // forest copies it with.
    TreeGaussianForestConfig forest;
    std::vector<float> heights;
    std::cout << "building the library copies ..." << std::endl;
    const std::vector<GaussianLods> library =
        buildTreeGaussianLibraryLevels({scene.types[type]}, {kSeason}, heights, forest.variants, forest.budgets);
    const uint32_t variant = tree::treeSiteSeed(scene.config.seed, yaw, forest.variants) - 1u;

    HeadlessFrontend frontend;
    VulkanContext& vc = frontend.getKlartraumEngine().getVulkanContext();
    const VkExtent2D extent{width, height};
    auto camera = std::make_shared<CameraUboType>();
    TreeView generated(vc, scene, site, false, extent, camera, library, variant, forest, false, leafLod, stemRound,
                       stemFade);
    TreeView copy(vc, scene, site, true, extent, camera, library, variant, forest);
    TreeView reference(vc, scene, site, false, extent, camera, library, variant, forest, true);

    const VkExtent2D panel{width * 3 / 4, height * 3 / 4};
    std::filesystem::create_directories(outDir / "frames");
    std::ofstream csv(outDir / "metrics.csv");
    csv << "frame,distance_m,drawn_by,level,fade,crown_px,residual_static,residual_dynamic,residual_free_static,"
           "residual_free_dynamic,fidelity_static,fidelity_dynamic,gpu_ms_static,gpu_ms_dynamic,gpu_ms_free_static,"
           "gpu_ms_free_dynamic\n";

    const glm::vec3 middle = glm::vec3(site.posHeight) + glm::vec3(0.0f, 0.5f * treeHeight, 0.0f);
    const glm::vec4 wind(glm::normalize(glm::vec3(1.0f, 0.0f, 0.35f)), windStrength);
    // The runs: 0 static, 1 dynamic, 2 static without levels of detail, 3 dynamic without.
    std::vector<PoppingMeter> poppingMeters(4, PoppingMeter(width, height));
    std::vector<std::vector<float>> residuals(4, std::vector<float>(size_t(frames) + 1, 0.0f));
    std::vector<std::vector<float>> fidelity(2, std::vector<float>(size_t(frames) + 1, 0.0f));
    std::vector<std::vector<float>> gpuMs(4, std::vector<float>(size_t(frames) + 1, 0.0f));
    std::vector<float> distances(size_t(frames) + 1);
    std::vector<std::string> drawnBy(size_t(frames) + 1);
    std::vector<glm::vec3> lods(size_t(frames) + 1); // level, fade, crown pixels
    std::vector<uint8_t> rgb(size_t(panel.width) * 2 * panel.height * 2 * 3);
    std::printf("tree %.1f m tall (%s, variant %u), generated nearer than %.1f m, %ux%u at %.0f deg, %d frames\n",
                treeHeight / m, species.c_str(), variant + 1, nearRadiusMeters, width, height, fov, frames);
    for (int k = 0; k <= frames; ++k) {
        // The logarithm of the distance along a cosine: out to --far and back.
        const float u = 0.5f - 0.5f * std::cos(2.0f * glm::pi<float>() * float(k) / float(frames));
        const float meters = startMeters * std::pow(farMeters / startMeters, u);
        const float d = meters * m;
        const glm::vec3 eye = middle + glm::vec3(0.0f, 0.0f, d);
        camera->ubo.model = glm::mat4(1.0f);
        camera->ubo.view = glm::lookAt(eye, middle, glm::vec3(0.0f, 1.0f, 0.0f));
        camera->ubo.proj =
            glm::perspective(glm::radians(fov), float(width) / float(height), 0.02f * d, 10.0f * d + 2.0f * treeHeight);
        camera->ubo.proj[1][1] *= -1.0f;
        camera->ubo.cameraWorldPos = glm::vec4(eye, 1.0f);
        camera->update(0);

        // As BonsaiRenderer: generated while among the near trees, else the copy at its level by its crown's size.
        const bool near = meters < nearRadiusMeters;
        const float focal = std::abs(camera->ubo.proj[1][1]) * 0.5f * float(height);
        const float pixels = tree::treePixelDiameter(eye, glm::vec3(site.posHeight), copy.crownRadius, focal);
        const glm::vec2 blend = tree::treeLodBlend(pixels, glm::vec3(forest.boundaries), forest.boundaries.w);
        char label[64];
        if (near) {
            std::snprintf(label, sizeof(label), "generated");
        } else if (blend.y > 0.0f) {
            std::snprintf(label, sizeof(label), "copy L%d>L%d", int(blend.x), int(blend.x) + 1);
        } else {
            std::snprintf(label, sizeof(label), "copy L%d", int(blend.x));
        }
        distances[k] = meters;
        drawnBy[k] = label;
        lods[k] = glm::vec3(near ? -1.0f : blend.x, near ? 0.0f : blend.y, pixels);

        const float seconds = float(k) / 30.0f;
        TreeView& view = near ? generated : copy;
        std::vector<uint8_t> images[4];
        for (int run = 0; run < 4; ++run) {
            TreeView& drawing = run < 2 ? view : reference;
            images[run] = drawing.render(seconds, run % 2 == 0 ? glm::vec4(glm::vec3(wind), 0.0f) : wind);
            gpuMs[run][k] = drawing.gpuMs;
        }
        for (int run = 0; run < 4; ++run) {
            if (const std::optional<float> r = poppingMeters[run].push(imageLuminance(images[run]))) {
                residuals[run][k - 1] = *r;
            }
        }
        fidelity[0][k] = fidelityError(images[0], images[2], width, height);
        fidelity[1][k] = fidelityError(images[1], images[3], width, height);

        // With levels of detail above, without below; static left, dynamic right.
        for (int run = 0; run < 4; ++run) {
            resampleInto(images[run], extent, rgb.data() + size_t(run / 2) * panel.height * panel.width * 2 * 3,
                         panel.width * 2, (run % 2) * panel.width, panel);
        }
        char name[32];
        std::snprintf(name, sizeof(name), "frame_%04d.ppm", k);
        std::ofstream file(outDir / "frames" / name, std::ios::binary);
        file << "P6\n" << panel.width * 2 << " " << panel.height * 2 << "\n255\n";
        file.write(reinterpret_cast<const char*>(rgb.data()), std::streamsize(rgb.size()));
        if (k % 100 == 0) {
            std::printf("frame %d: %.1f m, %s\n", k, meters, label);
        }
    }
    for (int k = 0; k <= frames; ++k) {
        csv << k << "," << distances[k] << "," << drawnBy[k] << "," << lods[k].x << "," << lods[k].y << "," << lods[k].z
            << "," << residuals[0][k] << "," << residuals[1][k] << "," << residuals[2][k] << "," << residuals[3][k]
            << "," << fidelity[0][k] << "," << fidelity[1][k] << "," << gpuMs[0][k] << "," << gpuMs[1][k] << ","
            << gpuMs[2][k] << "," << gpuMs[3][k] << "\n";
    }
    printPopping("static ", residuals[0], residuals[2], distances, drawnBy);
    printPopping("dynamic", residuals[1], residuals[3], distances, drawnBy);
    printByRepresentation("static  fidelity error", fidelity[0], distances, drawnBy);
    printByRepresentation("dynamic fidelity error", fidelity[1], distances, drawnBy);
    printByRepresentation("dynamic GPU ms", gpuMs[1], distances, drawnBy);
    printByRepresentation("dynamic GPU ms without levels of detail", gpuMs[3], distances, drawnBy);
    std::cout << "frames and metrics.csv in " << outDir.string() << std::endl;
    return 0;
}
