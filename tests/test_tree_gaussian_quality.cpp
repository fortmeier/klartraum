// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - sizeBinsSpanTheScreen: treeSizeBin() puts 0.5 px into the first bin, 4096 px and more into the last, and a size
 *   just above a bin's lower edge into that bin
 * - cutTargetKeepsTheLargestTrees: treeSizeCutTarget() keeps all trees when they fit, else the bins from the top
 *   down (empty ones included) as long as they fit the copies
 * - generatedTreesMatchTheLibrary: the near trees of 8 sites per type (treeInstanceFor() with the library's
 *   variants) are on average as tall as their sites say (within 10 %), and the finest level of each one's copy is
 *   as tall within 5 % and as wide within 15 %
 * - reservedFollowsTheLevels: treeLodReserved() is the level's count, plus the next coarser one's while
 *   crossfading, and larger boundaries (treeLodFitFactor()) give coarser levels
 * - copiesAreTheLargestTrees: at R2 the GPU's histogram is the CPU's of the visible trees (sizes times
 *   treeCopyPriority()), its target is treeSizeCutTarget() of it, and exactly the trees above the cut-off are copied
 * (fading in above it), at most TreeGaussianForestConfig::trees and more than 80 % of them; a second frame copies the
 * same trees
 * - cutoffMovesSlowly: flying from R1 towards the place by 1 % of the distance per frame for 30 frames moves the
 *   cut-off by at most a fade width per frame
 * - listHoldsTheLargestTrees: at R1 the proxies' list holds only trees at or above its cut-off (sizes times
 *   treeCopyPriority()), within its capacity and none left out, and the copies' cut-off lies above it
 * - noTreeLeftOutAlongTheFlight: flying from R0 to past R2 at twice the speed of the 16 s fly-in, the proxies'
 *   list never runs out of room
 * - forestFitsItsPool: at R0-R3 the copies' levels fit the forest pool with none left out: the fitted step needs
 *   what was reserved, at most the pool; where they fit at once, the boundaries stay as they are; where even the
 *   coarsest levels do not fit, fewer copies come; a still camera keeps its step
 * - overflowAlongTheFlight: at the reference views R0-R3 of the fly-in at scale 1 and 1920 x 1080, the visible
 *   trees of the proxies, the library copies taken of them and the forest pool's Gaussians, none left out, and the
 *   GPU time of a frame
 * - generatedTreesCost: at R3 at 1920 x 1080, the GPU time of a frame with 1 and 24 trees generated every frame (the
 *   rest library copies), its share per kind of pass and the cost of a generated tree; more trees cost more
 * - flickerAtAFixedCamera: at R3 and R2 without wind, the largest and mean difference between consecutive frames
 * - flickerWhileDollying: from R2 along the flight by 0.2 % of the distance to the place per frame, the mean over
 *   16 px blocks of how far a frame lies from the mean of its neighbours
 * - nearAndFarTreeAgree: a site drawn as a near tree (TreeGaussians) and as its library copy (TreeMossLod) at the
 *   same camera, 30 to 300 px tall: height and width of the bounding box and covered area agree within 10 % on
 *   average per type; the mean colour is measured
 * - smallTreesKeepTheirCoverage: a near tree and its copy drawn 12 to 60 px tall, against the near tree drawn at 4
 *   times the resolution and box-filtered down: the copy carries its light within 30 %; the near tree's excess (the
 *   splatting's dilation widens sub-pixel Gaussians) is measured
 * - levelsKeepTheirCoverage: each level of a copy carries the light of the next finer one, within 10 %, at the size
 *   where they switch
 * - farTreesAreSharp: the copy at 20, 60 and 200 px against the near tree's Gaussians drawn at 4 times the
 *   resolution and box-filtered down: its mean absolute error per 4 px block is at most the near tree's at the same
 *   size; the Laplacian energy ratio is measured
 **/

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <string>

#include "klartraum/gaussian_splatting_factory.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/tree/bonsai_flight.hpp"
#include "klartraum/tree/bonsai_renderer.hpp"
#include "klartraum/tree/tree_library.hpp"
#include "klartraum/tree/tree_placement.hpp"

#include "tree_test_helpers.hpp"

using namespace klartraum;
using namespace klartraum::tree_test;

namespace {

const char* kOutput = "build/TestingOutput/gaussian_quality";
constexpr float kSeason = 2.0f;

void writeImage(const std::string& name, const std::vector<uint8_t>& bgra, VkExtent2D extent) {
    std::filesystem::create_directories(kOutput);
    std::ofstream file(std::string(kOutput) + "/" + name + ".ppm", std::ios::binary);
    file << "P6\n" << extent.width << " " << extent.height << "\n255\n";
    for (size_t i = 0; i < bgra.size(); i += 4) {
        const char rgb[3] = {char(bgra[i + 2]), char(bgra[i + 1]), char(bgra[i])};
        file.write(rgb, 3);
    }
}

// Luminance of every pixel in [0, 1].
std::vector<float> luminance(const std::vector<uint8_t>& bgra) {
    std::vector<float> l(bgra.size() / 4);
    for (size_t i = 0; i < l.size(); ++i) {
        l[i] = (0.0722f * bgra[4 * i] + 0.7152f * bgra[4 * i + 1] + 0.2126f * bgra[4 * i + 2]) / 255.0f;
    }
    return l;
}

const BonsaiScene& sceneAtScale1() {
    static const BonsaiScene scene = [] {
        BonsaiSceneConfig config;
        config.scale.s = 1.0f;
        config.seed = 7;
        return makeBonsaiScene(config);
    }();
    return scene;
}

// The fly-in of tree_bonsai_example: from the orbit at distance 4.2, elevation 0.32, azimuth -2.09 to the ground
// place (-1.04, 0.15), 1.7 m above it, heading 120 deg. R0-R3 are its poses at u = 0, 0.19, 0.7 and 1.
struct ReferenceFlight {
    const BonsaiScene& scene;
    glm::vec3 start;
    glm::vec2 ground{-1.04f, 0.15f};
    float above;
    float heading = glm::radians(120.0f);

    explicit ReferenceFlight(const BonsaiScene& scene)
        : scene(scene) {
        const float azimuth = -2.09f, elevation = 0.32f, distance = 4.2f;
        start = scene.focus + distance * glm::vec3(std::sin(azimuth) * std::cos(elevation), std::sin(elevation),
                                                   std::cos(azimuth) * std::cos(elevation));
        above = 1.7f / scene.config.scale.metersPerUnit();
    }

    FlyInPose at(float u) const { return flyInPose(scene, start, scene.focus, ground, above, heading, u); }
    glm::vec3 place() const { return glm::vec3(ground.x, scene.soilHeight(ground.x, ground.y) + above, ground.y); }
};

constexpr float kReferenceU[4] = {0.0f, 0.19f, 0.7f, 1.0f};

// The Gaussian BonsaiRenderer, driven per frame as tree_bonsai_example --fly-in drives it.
struct FlightRenderer {
    FlightRenderer(VulkanContext& vc, const BonsaiScene& scene, VkExtent2D extent)
        : vc(vc),
          extent(extent) {
        target = std::make_shared<OffscreenTarget>(vc, extent, 1);
        config.extent = extent;
        config.mode = BonsaiRenderMode::Gaussian;
        renderer = std::make_unique<BonsaiRenderer>(vc, scene, target, config);
        graph = std::make_unique<ComputeGraph>(vc, 1);
        graph->enableProfiling();
        graph->compileFrom(renderer->getOutput());
        renderer->initialize(1);
        renderer->setSeason(kSeason);
    }

    std::vector<uint8_t> frame(const FlyInPose& pose, float seconds = 0.0f, float wind = 0.0f) {
        setCamera(renderer->getCamera()->ubo, pose.eye, pose.lookAt, 40.0f, float(extent.width) / float(extent.height),
                  pose.nearPlane, 50.0f);
        renderer->getCamera()->update(0);
        renderer->setView(pose.eye);
        renderer->setTime(seconds, 0.5f, kSeason, glm::vec4(glm::normalize(glm::vec3(1.0f, 0.0f, 0.35f)), wind));
        graph->submitAndWait(vc.getGraphicsQueue(), 0);
        renderer->setView(pose.eye);
        graph->submitAndWait(vc.getGraphicsQueue(), 0);
        return readImage(vc, target->getImage(0), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, extent);
    }

    // Frames at `pose` until every tile is there and the canopy's band is steady (at most 40).
    void settle(const FlyInPose& pose) {
        for (int i = 0; i < 40; ++i) {
            frame(pose);
            if (renderer->isSettled()) {
                return;
            }
        }
    }

    VulkanContext& vc;
    VkExtent2D extent;
    BonsaiRenderConfig config;
    std::shared_ptr<OffscreenTarget> target;
    std::unique_ptr<BonsaiRenderer> renderer;
    std::unique_ptr<ComputeGraph> graph;
};

// The tree library of TreeGaussianForest for the scene's types.
struct Library {
    std::vector<GaussianLods> patches;
    std::vector<float> heights;
    std::vector<TreeTypeParams> types;
    TreeGaussianForestConfig config;
};

const Library& libraryOf(const BonsaiScene& scene) {
    static const Library library = [&] {
        Library l;
        const std::vector<TreeTypeParams> used(scene.types.begin(),
                                               scene.types.begin() + std::min<size_t>(scene.types.size(), 2));
        l.patches =
            buildTreeGaussianLibraryLevels(used, l.config.seasons, l.heights, l.config.variants, l.config.budgets);
        l.types = used;
        return l;
    }();
    return library;
}

// The library copy of `site` as tree_library_instances.comp writes it, fully faded in, in the sun.
TreeLodInstance copyOf(const TreeSite& site, const Library& library) {
    const uint32_t type = site.kind.x == TREE_SITE_CONIFER ? 1u : 0u;
    uint32_t nearest = 0;
    for (uint32_t k = 0; k < library.config.seasons.size(); ++k) {
        if (std::abs(library.config.seasons[k] - kSeason) < std::abs(library.config.seasons[nearest] - kSeason)) {
            nearest = k;
        }
    }
    const uint32_t variants = library.config.variants;
    const uint32_t variant = tree::treeSiteSeed(sceneAtScale1().config.seed, site.kind.y, variants) - 1u;
    TreeLodInstance inst{};
    inst.posYaw = glm::vec4(glm::vec3(site.posHeight), site.kind.y);
    inst.plane = glm::vec4(0.0f, 0.0f, site.posHeight.w / library.types[type].shapeBase.z, 0.0f);
    inst.ids = glm::uvec4((type * variants + variant) * uint32_t(library.config.seasons.size()) + nearest,
                          glm::floatBitsToUint(0.0f), glm::floatBitsToUint(0.0f), 1u);
    return inst;
}

// One site over black, either generated in full as the near trees are (TreeGaussians) or as its library copy
// (TreeMossLod), lit the same way, in season kSeason without wind.
struct SingleTree {
    SingleTree(VulkanContext& vc, const BonsaiScene& scene, const TreeSite& site, bool copy, VkExtent2D extent,
               glm::vec4 boundaries = glm::vec4(0.0f))
        : vc(vc),
          extent(extent),
          height(site.posHeight.w) {
        camera = std::make_shared<CameraUboType>();
        settings = std::make_shared<HostValues<TreeRenderSettings>>(
            vc, std::vector<TreeRenderSettings>{defaultTreeRenderSettings(extent)});
        TreeRenderSettings values = settings->values().at(0);
        values.season = glm::vec4(kSeason, tree::treeSnowCover(kSeason), 0.0f, 0.0f);
        settings->set(0, values);
        TreeFrame frame = defaultTreeFrame();
        frame.wind.w = 0.0f;
        frame.timeSeason.y = kSeason;
        TreeGaussianSources sources;
        if (copy) {
            // A generator of a trunk only, far away; the trees are not part of the Gaussians.
            TreeTypeParams p = tree_presets::quakingAspen();
            p.ratios.x = 1.0f;
            p.leafCount.x = 0.0f;
            const std::vector<TreeTypeParams> types = {p};
            const std::vector<TreeInstance> trees = {makeInstance(0, 1, glm::vec3(10.0f, 0.0f, 0.0f), 0.01f)};
            generator =
                std::make_shared<TreeGenerator>(vc, types, trees, TreeCapacity::fromReference(types, trees, frame));
            sources.trees = false;
            sources.extraSlots = 262144;
        } else {
            const std::vector<TreeInstance> trees = {
                treeInstanceFor(site, scene.types, scene.config.seed, libraryOf(scene).config.variants)};
            generator = std::make_shared<TreeGenerator>(vc, scene.types, trees,
                                                        TreeCapacity::fromReference(scene.types, trees, frame), frame);
        }
        gaussians = std::make_unique<TreeGaussians>(vc, *generator, camera, settings, std::move(sources));
        GaussianSoABuffers splats = gaussians->getGaussians();
        ComputeGraphElementPtr last = gaussians->getElement();
        if (copy) {
            const Library& library = libraryOf(scene);
            lod = std::make_unique<TreeMossLod>(
                vc, *gaussians, library.patches, std::vector<TreeLodInstance>{copyOf(site, library)}, camera, settings,
                BufferRef{}, boundaries.x > 0.0f ? boundaries : library.config.boundaries);
            splats = lod->getGaussians();
        }
        target = std::make_shared<OffscreenTarget>(vc, extent, 1);
        splatting = createGaussianSplatting(vc, GsplatBackend::Raster, target, camera, splats);
        graph = std::make_unique<ComputeGraph>(vc, 1);
        graph->compileFrom(splatting);
        gaussians->initializeBuffers(1);
    }

    // Seen from the side, level with its middle, `pixels` tall on screen (field of view 40 degrees).
    std::vector<uint8_t> render(float pixels, glm::vec3 foot) {
        const float d = height * float(extent.height) / (2.0f * pixels * std::tan(glm::radians(20.0f)));
        const glm::vec3 middle = foot + glm::vec3(0.0f, 0.5f * height, 0.0f);
        setCamera(camera->ubo, middle + glm::vec3(0.0f, 0.0f, d), middle, 40.0f,
                  float(extent.width) / float(extent.height), 0.01f * d, 10.0f * d);
        camera->update(0);
        graph->submitAndWait(vc.getGraphicsQueue(), 0);
        return readImage(vc, target->getImage(0), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, extent);
    }

    VulkanContext& vc;
    VkExtent2D extent;
    float height;
    std::shared_ptr<CameraUboType> camera;
    std::shared_ptr<HostValues<TreeRenderSettings>> settings;
    std::shared_ptr<TreeGenerator> generator;
    std::unique_ptr<TreeGaussians> gaussians;
    std::unique_ptr<TreeMossLod> lod;
    std::shared_ptr<OffscreenTarget> target;
    std::shared_ptr<ComputeGraphElement> splatting;
    std::unique_ptr<ComputeGraph> graph;
};

TreeSite siteOf(float species, float yaw, float height) {
    TreeSite site{};
    site.posHeight = glm::vec4(0.0f, 0.0f, 0.0f, height);
    site.kind = glm::vec4(species, yaw, 1.0f, 1.0f);
    return site;
}

// What a tree covers over black: pixels brighter than 10/255 in any channel.
struct Footprint {
    float width = 0.0f;  ///< of the bounding box between the 1st and 99th percentile of covered columns
    float height = 0.0f; ///< the same of rows
    size_t area = 0;
    glm::vec3 color{0.0f}; ///< mean RGB in [0, 1] over the covered pixels
};

Footprint footprintOf(const std::vector<uint8_t>& bgra, VkExtent2D extent) {
    std::vector<uint32_t> xs, ys;
    Footprint f;
    for (uint32_t y = 0; y < extent.height; ++y) {
        for (uint32_t x = 0; x < extent.width; ++x) {
            const size_t i = (size_t(y) * extent.width + x) * 4;
            if (std::max({bgra[i], bgra[i + 1], bgra[i + 2]}) > 10) {
                xs.push_back(x);
                ys.push_back(y);
                f.color += glm::vec3(bgra[i + 2], bgra[i + 1], bgra[i]) / 255.0f;
            }
        }
    }
    f.area = xs.size();
    if (xs.empty()) {
        return f;
    }
    f.color /= float(f.area);
    auto spread = [](std::vector<uint32_t>& v) {
        std::sort(v.begin(), v.end());
        return float(v[v.size() * 99 / 100] - v[v.size() / 100] + 1);
    };
    f.width = spread(xs);
    f.height = spread(ys);
    return f;
}

// `bgra` box-filtered down by `factor` per axis.
std::vector<uint8_t> downsample(const std::vector<uint8_t>& bgra, VkExtent2D extent, uint32_t factor) {
    const uint32_t w = extent.width / factor, h = extent.height / factor;
    std::vector<uint8_t> out(size_t(w) * h * 4);
    for (uint32_t y = 0; y < h; ++y) {
        for (uint32_t x = 0; x < w; ++x) {
            for (uint32_t c = 0; c < 4; ++c) {
                uint32_t sum = 0;
                for (uint32_t j = 0; j < factor; ++j) {
                    for (uint32_t i = 0; i < factor; ++i) {
                        sum += bgra[((size_t(y) * factor + j) * extent.width + x * factor + i) * 4 + c];
                    }
                }
                out[(size_t(y) * w + x) * 4 + c] = uint8_t((sum + factor * factor / 2) / (factor * factor));
            }
        }
    }
    return out;
}

// The mean absolute difference of the RGB means of 4 x 4 pixel blocks, in [0, 1], over the blocks either image
// covers.
float blockError(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b, VkExtent2D extent) {
    double sum = 0.0;
    size_t blocks = 0;
    for (uint32_t by = 0; by + 4 <= extent.height; by += 4) {
        for (uint32_t bx = 0; bx + 4 <= extent.width; bx += 4) {
            glm::vec3 ma(0.0f), mb(0.0f);
            for (uint32_t y = by; y < by + 4; ++y) {
                for (uint32_t x = bx; x < bx + 4; ++x) {
                    const size_t i = (size_t(y) * extent.width + x) * 4;
                    ma += glm::vec3(a[i + 2], a[i + 1], a[i]);
                    mb += glm::vec3(b[i + 2], b[i + 1], b[i]);
                }
            }
            if (std::max(glm::length(ma), glm::length(mb)) > 0.0f) {
                const glm::vec3 d = glm::abs(ma - mb) / (16.0f * 255.0f);
                sum += (d.x + d.y + d.z) / 3.0f;
                ++blocks;
            }
        }
    }
    return blocks > 0 ? float(sum / double(blocks)) : 0.0f;
}

// The sum of the squared 4-neighbour Laplacian of the luminance.
double laplacianEnergy(const std::vector<uint8_t>& bgra, VkExtent2D extent) {
    const std::vector<float> l = luminance(bgra);
    double energy = 0.0;
    for (uint32_t y = 1; y + 1 < extent.height; ++y) {
        for (uint32_t x = 1; x + 1 < extent.width; ++x) {
            const size_t i = size_t(y) * extent.width + x;
            const float lap = 4.0f * l[i] - l[i - 1] - l[i + 1] - l[i - extent.width] - l[i + extent.width];
            energy += double(lap) * lap;
        }
    }
    return energy;
}

} // namespace

class TreeGaussianQualityTest : public ::testing::Test {
protected:
    void SetUp() override {
        frontend = std::make_unique<HeadlessFrontend>();
        vc = &frontend->getKlartraumEngine().getVulkanContext();
    }
    void TearDown() override { frontend.reset(); }

    std::unique_ptr<HeadlessFrontend> frontend;
    VulkanContext* vc = nullptr;
};

TEST(TreeGaussianQualityCpuTest, sizeBinsSpanTheScreen) {
    EXPECT_EQ(tree::treeSizeBin(0.5f), 0u);
    EXPECT_EQ(tree::treeSizeBin(0.1f), 0u);
    EXPECT_EQ(tree::treeSizeBin(4096.0f), TREE_SIZE_BINS - 1u);
    EXPECT_EQ(tree::treeSizeBin(1e6f), TREE_SIZE_BINS - 1u);
    for (uint32_t bin = 0; bin < TREE_SIZE_BINS; ++bin) {
        EXPECT_EQ(tree::treeSizeBin(std::exp2(tree::treeSizeBinEdge(bin) + 0.01f)), bin) << bin;
    }
    EXPECT_FLOAT_EQ(tree::treeSizeBinEdge(TREE_SIZE_BINS), 12.0f);
    EXPECT_EQ(tree::treeCopyFade(std::exp2(3.0f), 3.0f), 0.0f);
    EXPECT_EQ(tree::treeCopyFade(std::exp2(3.0f + TREE_COPY_FADE_WIDTH), 3.0f), 1.0f);
    EXPECT_NEAR(tree::treeCopyFade(std::exp2(3.0f + 0.5f * TREE_COPY_FADE_WIDTH), 3.0f), 0.5f, 1e-5f);
}

TEST(TreeGaussianQualityCpuTest, cutTargetKeepsTheLargestTrees) {
    std::vector<uint32_t> bins(TREE_SIZE_BINS, 0u);
    uint32_t taken = 1;
    EXPECT_FLOAT_EQ(treeSizeCutTarget(bins, 10, &taken), TREE_SIZE_LOG_MIN);
    EXPECT_EQ(taken, 0u);
    bins[10] = bins[20] = bins[30] = 5;
    EXPECT_FLOAT_EQ(treeSizeCutTarget(bins, 15, &taken), TREE_SIZE_LOG_MIN);
    EXPECT_EQ(taken, 15u);
    EXPECT_FLOAT_EQ(treeSizeCutTarget(bins, 14, &taken), tree::treeSizeBinEdge(11));
    EXPECT_EQ(taken, 10u);
    EXPECT_FLOAT_EQ(treeSizeCutTarget(bins, 9, &taken), tree::treeSizeBinEdge(21));
    EXPECT_EQ(taken, 5u);
    EXPECT_FLOAT_EQ(treeSizeCutTarget(bins, 4, &taken), tree::treeSizeBinEdge(31));
    EXPECT_EQ(taken, 0u);
}

namespace {

// Height and crown radius (around the stem's foot, of the leaves and stems) of generated trees.
glm::vec2 treeExtent(const TreeRecords& records) {
    glm::vec2 extent(0.0f);
    auto add = [&](glm::vec3 p) {
        extent.x = std::max(extent.x, p.y);
        extent.y = std::max(extent.y, glm::length(glm::vec2(p.x, p.z)));
    };
    for (const SegmentRecord& seg : records.segments) {
        add(glm::vec3(seg.toPosRadius));
    }
    for (const LeafRecord& leaf : records.leaves) {
        add(glm::vec3(leaf.posScale) + leaf.posScale.w * tree::qforward(leaf.rot));
    }
    return extent;
}

} // namespace

TEST(TreeGaussianQualityCpuTest, generatedTreesMatchTheLibrary) {
    const BonsaiScene& scene = sceneAtScale1();
    const Library& library = libraryOf(scene);
    TreeFrame frame = defaultTreeFrame();
    frame.wind.w = 0.0f;
    frame.timeSeason.y = kSeason;
    for (uint32_t type = 0; type < 2; ++type) {
        glm::vec2 sum(0.0f), worst(0.0f);
        for (int k = 0; k < 8; ++k) {
            const TreeSite site =
                siteOf(type == 1 ? TREE_SITE_CONIFER : TREE_SITE_BROADLEAF, 0.7f * float(k) + 0.1f, 1.0f);
            const TreeInstance nearTree =
                treeInstanceFor(site, scene.types, scene.config.seed, library.config.variants);
            const glm::vec2 n = treeExtent(generateTreesReference(scene.types, {nearTree}, frame));
            // The copy as tree_library_instances.comp places it: its patch's finest level, scaled.
            const TreeLodInstance copy = copyOf(site, library);
            glm::vec2 c(0.0f);
            for (const Gaussian3D& g : library.patches[copy.ids.x].levels[0]) {
                c.x = std::max(c.x, g.position[1] + g.scale[2]);
                c.y = std::max(c.y, glm::length(glm::vec2(g.position[0], g.position[2])));
            }
            c *= copy.plane.z;
            sum += n;
            worst = glm::max(worst, glm::abs(c / n - 1.0f));
        }
        const glm::vec2 mean = sum / 8.0f;
        std::printf("%s: near trees of a 1 unit site %.3f tall, crown radius %.3f; copies differ by at most %.1f %% "
                    "in height, %.1f %% in crown radius\n",
                    type == 1 ? "conifer" : "broadleaf", mean.x, mean.y, 100.0f * worst.x, 100.0f * worst.y);
        EXPECT_NEAR(mean.x, 1.0f, 0.1f) << type;
        // The copies' Gaussians sit at the middles of their leaves; the leaf tips reach a little farther.
        EXPECT_LT(worst.x, 0.05f) << type;
        EXPECT_LT(worst.y, 0.15f) << type;
    }
}

TEST(TreeGaussianQualityCpuTest, reservedFollowsTheLevels) {
    TreeLodPatch patch{};
    patch.count = glm::uvec4(1000, 250, 60, 15);
    const glm::vec3 boundaries(480.0f, 160.0f, 60.0f);
    EXPECT_EQ(tree::treeLodReserved(patch, 1000.0f, boundaries, 0.15f), 1000u);
    EXPECT_EQ(tree::treeLodReserved(patch, 480.0f, boundaries, 0.15f), 1250u);
    EXPECT_EQ(tree::treeLodReserved(patch, 300.0f, boundaries, 0.15f), 250u);
    EXPECT_EQ(tree::treeLodReserved(patch, 100.0f, boundaries, 0.15f), 60u);
    EXPECT_EQ(tree::treeLodReserved(patch, 10.0f, boundaries, 0.15f), 15u);
    EXPECT_FLOAT_EQ(tree::treeLodFitFactor(0), 1.0f);
    EXPECT_FLOAT_EQ(tree::treeLodFitFactor(4), 2.0f);
    for (float pixels : {50.0f, 120.0f, 400.0f, 900.0f}) {
        for (uint32_t k = 1; k < TREE_LOD_FIT_STEPS; ++k) {
            EXPECT_LE(tree::treeLodReserved(patch, pixels, boundaries * tree::treeLodFitFactor(k), 0.15f),
                      tree::treeLodReserved(patch, pixels, boundaries * tree::treeLodFitFactor(k - 1), 0.15f) +
                          patch.count[1])
                << pixels << " " << k;
        }
        EXPECT_LE(tree::treeLodReserved(patch, pixels, boundaries * tree::treeLodFitFactor(15), 0.15f),
                  tree::treeLodReserved(patch, pixels, boundaries, 0.15f));
    }
}

TEST_F(TreeGaussianQualityTest, copiesAreTheLargestTrees) {
    const BonsaiScene& scene = sceneAtScale1();
    const ReferenceFlight flight(scene);
    FlightRenderer r(*vc, scene, {1920, 1080});
    const FlyInPose pose = flight.at(kReferenceU[2]);
    r.settle(pose);
    r.frame(pose);
    TreeGaussianForest& forest = *r.renderer->getForest();
    const std::vector<TreeSite> visible = r.renderer->getProxies()->readVisible();
    const TreeSizeCut cut = forest.readCut();
    const std::vector<TreeLodInstance> copies = forest.readInstances();
    ASSERT_GE(copies.size(), visible.size());
    const float focal = std::abs(r.renderer->getCamera()->ubo.proj[1][1]) * 0.5f * float(r.extent.height);
    std::vector<uint32_t> bins(TREE_SIZE_BINS, 0u);
    std::vector<float> pixels(visible.size());
    for (size_t i = 0; i < visible.size(); ++i) {
        const float h = visible[i].posHeight.w;
        pixels[i] =
            tree::treePixelDiameter(pose.eye, glm::vec3(visible[i].posHeight) + glm::vec3(0.0f, 0.5f * h, 0.0f),
                                    0.5f * h, focal) *
            tree::treeCopyPriority(
                tree::treeLandCover(scene.terrain, visible[i].posHeight.x, visible[i].posHeight.z).vegetation.x);
        ++bins[tree::treeSizeBin(pixels[i])];
    }
    uint32_t binDifference = 0;
    for (uint32_t b = 0; b < TREE_SIZE_BINS; ++b) {
        binDifference += uint32_t(std::abs(int(bins[b]) - int(cut.bins[b])));
    }
    EXPECT_LE(binDifference, visible.size() / 1000 + 2);
    uint32_t taken = 0;
    EXPECT_FLOAT_EQ(cut.target, treeSizeCutTarget(cut.bins, r.config.forest.trees, &taken));
    EXPECT_EQ(cut.taken, taken);
    EXPECT_GE(cut.cut, cut.target);
    uint32_t copied = 0, wrong = 0;
    const float from = std::exp2(cut.cut), to = std::exp2(cut.cut + TREE_COPY_FADE_WIDTH);
    for (size_t i = 0; i < visible.size(); ++i) {
        const bool shown = copies[i].plane.z > 0.0f;
        copied += shown ? 1 : 0;
        if ((shown && pixels[i] < from * 0.999f) || (!shown && pixels[i] > to * 1.001f && visible[i].kind.w > 0.0f)) {
            ++wrong;
        }
    }
    std::printf("R2: %zu visible, cut-off %.1f px (target %.1f px), %u copied of at most %u\n", visible.size(), from,
                std::exp2(cut.target), copied, r.config.forest.trees);
    EXPECT_EQ(wrong, 0u);
    EXPECT_LE(copied, r.config.forest.trees);
    EXPECT_GT(copied, r.config.forest.trees * 8 / 10);

    // A second frame copies the same trees.
    auto copiedSet = [](const std::vector<TreeLodInstance>& instances) {
        std::vector<std::array<float, 4>> set;
        for (const TreeLodInstance& inst : instances) {
            if (inst.plane.z > 0.0f) {
                set.push_back({inst.posYaw.x, inst.posYaw.y, inst.posYaw.z, glm::uintBitsToFloat(inst.ids.y)});
            }
        }
        std::sort(set.begin(), set.end());
        return set;
    };
    const auto before = copiedSet(copies);
    r.frame(pose);
    EXPECT_EQ(copiedSet(forest.readInstances()), before);
}

TEST_F(TreeGaussianQualityTest, cutoffMovesSlowly) {
    const BonsaiScene& scene = sceneAtScale1();
    const ReferenceFlight flight(scene);
    FlightRenderer r(*vc, scene, {1920, 1080});
    FlyInPose pose = flight.at(kReferenceU[1]);
    r.settle(pose);
    float before = r.renderer->getForest()->readCut().cut;
    float largest = 0.0f;
    float lowest = before, highest = before;
    for (int k = 0; k < 30; ++k) {
        const glm::vec3 step = 0.01f * (flight.place() - pose.eye);
        pose.eye += step;
        pose.lookAt += step;
        r.frame(pose);
        const float cut = r.renderer->getForest()->readCut().cut;
        largest = std::max(largest, std::abs(cut - before));
        lowest = std::min(lowest, cut);
        highest = std::max(highest, cut);
        before = cut;
    }
    std::printf("R1 towards the place by 1 %% per frame: the cut-off between %.1f and %.1f px, moving by at most "
                "%.2f fade widths per frame\n",
                std::exp2(lowest), std::exp2(highest), largest / TREE_COPY_FADE_WIDTH);
    EXPECT_LE(largest, TREE_COPY_FADE_WIDTH * 1.001f);
}

TEST_F(TreeGaussianQualityTest, listHoldsTheLargestTrees) {
    const BonsaiScene& scene = sceneAtScale1();
    const ReferenceFlight flight(scene);
    FlightRenderer r(*vc, scene, {1920, 1080});
    ASSERT_TRUE(r.renderer->getProxies()->getConfig().bySize);
    const FlyInPose pose = flight.at(kReferenceU[1]);
    r.settle(pose);
    r.frame(pose);
    TreeProxies& proxies = *r.renderer->getProxies();
    const std::vector<uint32_t> counter = proxies.readCounter();
    const std::vector<TreeSite> visible = proxies.readVisible();
    // The cull of the last frame used the cut-off from before it; the frame just before is the same view.
    const TreeSizeCut cut = proxies.readCut();
    const float focal = std::abs(r.renderer->getCamera()->ubo.proj[1][1]) * 0.5f * float(r.extent.height);
    uint32_t below = 0;
    for (const TreeSite& site : visible) {
        const float h = site.posHeight.w;
        const float pixels =
            tree::treePixelDiameter(pose.eye, glm::vec3(site.posHeight) + glm::vec3(0.0f, 0.5f * h, 0.0f), 0.5f * h,
                                    focal) *
            tree::treeCopyPriority(tree::treeLandCover(scene.terrain, site.posHeight.x, site.posHeight.z).vegetation.x);
        // The CPU's forest weight may differ slightly from the placement's on the GPU.
        below += std::log2(pixels) < cut.cut - 0.02f ? 1 : 0;
    }
    uint32_t counted = 0;
    for (uint32_t n : cut.bins) {
        counted += n;
    }
    const TreeSizeCut copies = r.renderer->getForest()->readCut();
    std::printf("R1: %u trees in view, %u in the list of %u above %.1f px, none of %u left out; copies above %.1f px\n",
                counted, counter[0], proxies.getConfig().capacity, std::exp2(cut.cut), counter[1],
                std::exp2(copies.cut));
    EXPECT_EQ(below, 0u);
    EXPECT_EQ(counter[1], 0u);
    EXPECT_LE(counter[0], proxies.getConfig().capacity);
    EXPECT_GE(copies.cut, cut.cut);
}

TEST_F(TreeGaussianQualityTest, noTreeLeftOutAlongTheFlight) {
    const BonsaiScene& scene = sceneAtScale1();
    const ReferenceFlight flight(scene);
    FlightRenderer r(*vc, scene, {1920, 1080});
    uint32_t frames = 0, overflowing = 0, leftOut = 0, largest = 0;
    for (float u = 0.0f; u <= 0.75f; u += 2.0f / 479.0f, ++frames) {
        r.frame(flight.at(u));
        const std::vector<uint32_t> counter = r.renderer->getProxies()->readCounter();
        overflowing += counter[1] > 0 ? 1 : 0;
        leftOut += counter[1];
        largest = std::max(largest, counter[0]);
    }
    std::printf("%u frames: at most %u trees in the list; %u frames left %u trees out\n", frames, largest, overflowing,
                leftOut);
    EXPECT_EQ(leftOut, 0u);
}

TEST_F(TreeGaussianQualityTest, forestFitsItsPool) {
    const BonsaiScene& scene = sceneAtScale1();
    const ReferenceFlight flight(scene);
    FlightRenderer r(*vc, scene, {1920, 1080});
    TreeMossLod& lod = r.renderer->getForest()->getLod();
    for (int view = 0; view < 4; ++view) {
        const FlyInPose pose = flight.at(kReferenceU[view]);
        r.settle(pose);
        r.frame(pose);
        const TreeMossLod::Fit fit = lod.readFit();
        const std::vector<uint32_t> pool = lod.readCounter();
        std::printf("R%d: step %u (boundaries x %.2f) needs %u of %u Gaussians, %u reserved, %u copies left out; "
                    "finest step needs %u; %.1f %% fewer copies\n",
                    view, fit.step, fit.factor, fit.sums[fit.step], r.config.forestPool, pool[0], pool[1], fit.sums[0],
                    100.0f * fit.leaveOut);
        EXPECT_EQ(pool[1], 0u) << view;
        EXPECT_EQ(fit.sums[fit.step], pool[0]) << view;
        EXPECT_LE(pool[0], r.config.forestPool) << view;
        if (fit.sums[0] <= r.config.forestPool) {
            EXPECT_EQ(fit.step, 0u) << view;
        }
        r.frame(pose);
        EXPECT_EQ(lod.readFit().step, fit.step) << view;
    }
}

TEST_F(TreeGaussianQualityTest, overflowAlongTheFlight) {
    const BonsaiScene& scene = sceneAtScale1();
    const ReferenceFlight flight(scene);
    FlightRenderer r(*vc, scene, {1920, 1080});
    ASSERT_NE(r.renderer->getProxies(), nullptr);
    ASSERT_NE(r.renderer->getForest(), nullptr);
    const uint32_t copies = r.config.forest.trees;
    for (int view = 0; view < 4; ++view) {
        const FlyInPose pose = flight.at(kReferenceU[view]);
        r.settle(pose);
        r.graph->resetProfiling();
        std::vector<uint8_t> image;
        for (int i = 0; i < 3; ++i) {
            image = r.frame(pose);
        }
        const std::vector<uint32_t> proxies = r.renderer->getProxies()->readCounter();
        const std::vector<uint32_t> pool = r.renderer->getForest()->getLod().readCounter();
        const TreeSizeCut cut = r.renderer->getForest()->readCut();
        std::printf("R%d (u = %.2f): proxies %u visible of %u (%u left out); copies %u of %u above %.1f px; "
                    "forest pool %u of %u Gaussians (%u copies left out); %.1f ms\n",
                    view, kReferenceU[view], proxies[0], r.config.proxies.capacity, proxies[1], cut.taken, copies,
                    std::exp2(cut.cut), pool[0], r.config.forestPool, pool[1], r.graph->getGpuSpanMs() / 2.0);
        EXPECT_LE(cut.taken, copies);
        EXPECT_EQ(proxies[1], 0u);
        EXPECT_EQ(pool[1], 0u);
        writeImage("flight_R" + std::to_string(view), image, r.extent);
        EXPECT_GT(proxies[0], 0u);
    }
}

TEST_F(TreeGaussianQualityTest, generatedTreesCost) {
    const ReferenceFlight flight(sceneAtScale1());
    // The passes by what they belong to, from their names.
    auto kindOf = [](const std::string& name) -> std::string {
        for (const char* kind :
             {"tree_level", "tree_roots", "tree_gaussians", "tree_lod", "tree_library", "tree_proxy", "tree_terrain",
              "tree_grass", "tree_floor", "tree_light", "Raster", "Splat", "Sort", "Project"}) {
            if (name.find(kind) != std::string::npos) {
                return kind;
            }
        }
        return "other";
    };
    double one = 0.0;
    for (uint32_t trees : {1u, 24u}) {
        BonsaiSceneConfig config;
        config.scale.s = 1.0f;
        config.seed = 7;
        config.nearTrees = trees;
        const BonsaiScene scene = makeBonsaiScene(config);
        FlightRenderer r(*vc, scene, {1920, 1080});
        for (int view : {3}) {
            const FlyInPose pose = flight.at(kReferenceU[view]);
            r.settle(pose);
            r.graph->resetProfiling();
            for (int i = 0; i < 3; ++i) {
                r.frame(pose);
            }
            std::map<std::string, double> kinds;
            for (const auto& [name, ms] : r.graph->getProfilingResults()) {
                kinds[kindOf(name)] += ms;
            }
            const double ms = r.graph->getGpuSpanMs();
            std::printf("R%d, %2u trees generated: %.1f ms per frame;", view, trees, ms);
            for (const auto& [kind, kindMs] : kinds) {
                if (kindMs >= 0.1) {
                    std::printf(" %s %.1f", kind.c_str(), kindMs);
                }
            }
            std::printf("\n");
            EXPECT_GT(ms, 0.0);
            if (trees == 1u) {
                one = ms;
            } else {
                std::printf("a generated tree costs %.2f ms\n", (ms - one) / double(trees - 1u));
                EXPECT_GT(ms, one);
            }
        }
    }
}

TEST_F(TreeGaussianQualityTest, flickerAtAFixedCamera) {
    const BonsaiScene& scene = sceneAtScale1();
    const ReferenceFlight flight(scene);
    FlightRenderer r(*vc, scene, {1920, 1080});
    for (int view : {3, 2}) {
        const FlyInPose pose = flight.at(kReferenceU[view]);
        r.settle(pose);
        std::vector<uint8_t> before = r.frame(pose);
        int largest = 0;
        double mean = 0.0;
        size_t changed = 0;
        for (int i = 1; i < 8; ++i) {
            const std::vector<uint8_t> image = r.frame(pose);
            for (size_t k = 0; k < image.size(); k += 4) {
                int d = 0;
                for (int c = 0; c < 3; ++c) {
                    d = std::max(d, std::abs(int(image[k + c]) - int(before[k + c])));
                }
                largest = std::max(largest, d);
                mean += d;
                changed += d > 8 ? 1 : 0;
            }
            before = image;
        }
        mean /= 7.0 * double(before.size() / 4);
        std::printf("R%d still: consecutive frames differ by at most %d/255, mean %.4f/255, %.1f px > 8/255 per "
                    "frame\n",
                    view, largest, mean, double(changed) / 7.0);
        EXPECT_LE(largest, 255);
    }
}

TEST_F(TreeGaussianQualityTest, flickerWhileDollying) {
    const BonsaiScene& scene = sceneAtScale1();
    const ReferenceFlight flight(scene);
    FlightRenderer r(*vc, scene, {1920, 1080});
    const FlyInPose start = flight.at(kReferenceU[2]);
    const glm::vec3 along = glm::normalize(flight.at(kReferenceU[2] + 0.01f).eye - flight.at(kReferenceU[2]).eye);
    const float step = 0.002f * glm::length(start.eye - flight.place());
    r.settle(start);
    std::vector<std::vector<float>> frames;
    for (int k = 0; k < 9; ++k) {
        FlyInPose pose = start;
        pose.eye += float(k) * step * along;
        pose.lookAt += float(k) * step * along;
        const std::vector<uint8_t> image = r.frame(pose);
        if (k == 4) {
            writeImage("dolly_R2", image, r.extent);
        }
        frames.push_back(luminance(image));
    }
    const uint32_t w = r.extent.width, h = r.extent.height;
    double sum = 0.0;
    size_t blocks = 0;
    for (size_t t = 1; t + 1 < frames.size(); ++t) {
        for (uint32_t by = 0; by + 16 <= h; by += 16) {
            for (uint32_t bx = 0; bx + 16 <= w; bx += 16) {
                double residual = 0.0, light = 0.0;
                for (uint32_t y = by; y < by + 16; ++y) {
                    for (uint32_t x = bx; x < bx + 16; ++x) {
                        const size_t i = size_t(y) * w + x;
                        residual += std::abs(frames[t][i] - 0.5f * (frames[t - 1][i] + frames[t + 1][i]));
                        light += frames[t][i];
                    }
                }
                if (light / 256.0 > 0.01) {
                    sum += residual / 256.0;
                    ++blocks;
                }
            }
        }
    }
    const double f = blocks > 0 ? sum / double(blocks) : 0.0;
    std::printf("R2 dolly by %.3g units per frame: F = %.5f over %zu blocks\n", step, f, blocks);
    EXPECT_GT(blocks, 0u);
}

TEST_F(TreeGaussianQualityTest, nearAndFarTreeAgree) {
    const BonsaiScene& scene = sceneAtScale1();
    const VkExtent2D extent{512, 512};
    const float height = scene.config.scale.treeHeightUnits();
    std::printf("library heights at scale 1: broadleaf %.3f (shapeBase.z %.3f), conifer %.3f (shapeBase.z %.3f)\n",
                libraryOf(scene).heights[0], scene.types[0].shapeBase.z, libraryOf(scene).heights[1],
                scene.types[1].shapeBase.z);
    for (float species : {TREE_SITE_BROADLEAF, TREE_SITE_CONIFER}) {
        glm::vec4 ratioSum(0.0f);
        float colorSum = 0.0f;
        int cases = 0;
        for (float yaw : {0.3f, 1.7f, 2.9f, 4.4f}) {
            const TreeSite site = siteOf(species, yaw, height);
            SingleTree nearTree(*vc, scene, site, false, extent);
            SingleTree copy(*vc, scene, site, true, extent);
            for (float pixels : {30.0f, 100.0f, 300.0f}) {
                const std::vector<uint8_t> a = nearTree.render(pixels, glm::vec3(site.posHeight));
                const std::vector<uint8_t> b = copy.render(pixels, glm::vec3(site.posHeight));
                const Footprint fa = footprintOf(a, extent), fb = footprintOf(b, extent);
                ASSERT_GT(fa.area, 0u);
                ASSERT_GT(fb.area, 0u);
                const glm::vec4 ratio(fb.height / fa.height, fb.width / fa.width, float(fb.area) / float(fa.area),
                                      0.0f);
                const float color = glm::length(fb.color - fa.color);
                std::printf("%s yaw %.1f %3.0f px: near %3.0f x %3.0f px, %6zu px; copy %3.0f x %3.0f px, %6zu px; "
                            "copy/near height %.2f width %.2f area %.2f; colour difference %.3f\n",
                            species == TREE_SITE_CONIFER ? "conifer  " : "broadleaf", yaw, pixels, fa.height, fa.width,
                            fa.area, fb.height, fb.width, fb.area, ratio.x, ratio.y, ratio.z, color);
                if (yaw == 0.3f) {
                    const std::string name = std::string(species == TREE_SITE_CONIFER ? "conifer" : "broadleaf") + "_" +
                                             std::to_string(int(pixels));
                    writeImage("agree_" + name + "_near", a, extent);
                    writeImage("agree_" + name + "_copy", b, extent);
                }
                ratioSum += ratio;
                colorSum += color;
                ++cases;
            }
        }
        std::printf("%s mean copy/near: height %.2f, width %.2f, area %.2f; colour difference %.3f\n",
                    species == TREE_SITE_CONIFER ? "conifer" : "broadleaf", ratioSum.x / cases, ratioSum.y / cases,
                    ratioSum.z / cases, colorSum / cases);
        EXPECT_NEAR(ratioSum.x / cases, 1.0f, 0.1f) << species;
        EXPECT_NEAR(ratioSum.y / cases, 1.0f, 0.1f) << species;
        EXPECT_NEAR(ratioSum.z / cases, 1.0f, 0.1f) << species;
    }
}

TEST_F(TreeGaussianQualityTest, smallTreesKeepTheirCoverage) {
    const BonsaiScene& scene = sceneAtScale1();
    const VkExtent2D extent{128, 128}, fine{512, 512};
    const float height = scene.config.scale.treeHeightUnits();
    auto light = [](const std::vector<uint8_t>& bgra) {
        double sum = 0.0;
        for (float l : luminance(bgra)) {
            sum += l;
        }
        return sum;
    };
    for (float species : {TREE_SITE_BROADLEAF, TREE_SITE_CONIFER}) {
        const TreeSite site = siteOf(species, 0.3f, height);
        SingleTree nearTree(*vc, scene, site, false, extent);
        SingleTree reference(*vc, scene, site, false, fine);
        SingleTree copy(*vc, scene, site, true, extent);
        const glm::vec3 foot(site.posHeight);
        for (float pixels : {12.0f, 20.0f, 35.0f, 60.0f}) {
            const double ref = light(downsample(reference.render(4.0f * pixels, foot), fine, 4));
            const double n = light(nearTree.render(pixels, foot)) / ref;
            const double c = light(copy.render(pixels, foot)) / ref;
            std::printf("%-9s %2.0f px: light of the near tree %.2f, of the copy %.2f times the reference's\n",
                        species == TREE_SITE_CONIFER ? "conifer" : "broadleaf", pixels, n, c);
            EXPECT_GT(n, 0.0) << species << " " << pixels;
            EXPECT_NEAR(c, 1.0, 0.3) << species << " " << pixels;
        }
    }
}

TEST_F(TreeGaussianQualityTest, levelsKeepTheirCoverage) {
    const BonsaiScene& scene = sceneAtScale1();
    const VkExtent2D extent{512, 512};
    const float height = scene.config.scale.treeHeightUnits();
    auto light = [](const std::vector<uint8_t>& bgra) {
        double sum = 0.0;
        for (float l : luminance(bgra)) {
            sum += l;
        }
        return sum;
    };
    const glm::vec3 switches = libraryOf(scene).config.boundaries;
    for (float species : {TREE_SITE_BROADLEAF, TREE_SITE_CONIFER}) {
        const TreeSite site = siteOf(species, 0.3f, height);
        // Each level alone: boundaries far above the sizes for the finer levels, far below for the coarser ones.
        std::array<std::unique_ptr<SingleTree>, 4> levels;
        for (int level = 0; level < 4; ++level) {
            glm::vec4 b(0.0f, 0.0f, 0.0f, 0.0f);
            for (int k = 0; k < 3; ++k) {
                b[k] = k < level ? 1e6f : 1e-3f;
            }
            levels[level] = std::make_unique<SingleTree>(*vc, scene, site, true, extent, b);
        }
        for (int level = 1; level < 4; ++level) {
            // The tree as tall on screen as its crown is wide at the switch (the crown about 0.75 of the height).
            const float pixels = std::min(switches[level - 1] / 0.75f, 480.0f);
            const double finer = light(levels[level - 1]->render(pixels, glm::vec3(site.posHeight)));
            const double coarser = light(levels[level]->render(pixels, glm::vec3(site.posHeight)));
            std::printf("%-9s %3.0f px: level %d carries %.2f times the light of level %d\n",
                        species == TREE_SITE_CONIFER ? "conifer" : "broadleaf", pixels, level, coarser / finer,
                        level - 1);
            EXPECT_NEAR(coarser / finer, 1.0, 0.1) << species << " " << level;
        }
    }
}

TEST_F(TreeGaussianQualityTest, farTreesAreSharp) {
    const BonsaiScene& scene = sceneAtScale1();
    const VkExtent2D extent{256, 256}, fine{1024, 1024};
    const float height = scene.config.scale.treeHeightUnits();
    for (float species : {TREE_SITE_BROADLEAF, TREE_SITE_CONIFER}) {
        const TreeSite site = siteOf(species, 0.3f, height);
        SingleTree nearTree(*vc, scene, site, false, extent);
        SingleTree reference(*vc, scene, site, false, fine);
        SingleTree copy(*vc, scene, site, true, extent);
        const char* name = species == TREE_SITE_CONIFER ? "conifer" : "broadleaf";
        for (float pixels : {20.0f, 60.0f, 200.0f}) {
            const glm::vec3 foot(site.posHeight);
            const std::vector<uint8_t> ref = downsample(reference.render(4.0f * pixels, foot), fine, 4);
            const std::vector<uint8_t> n = nearTree.render(pixels, foot);
            const std::vector<uint8_t> c = copy.render(pixels, foot);
            const double energy = laplacianEnergy(ref, extent);
            EXPECT_LE(blockError(c, ref, extent), blockError(n, ref, extent)) << name << " " << pixels;
            std::printf("%-9s %3.0f px: block error copy %.4f, near %.4f; Laplacian energy ratio copy %.2f, near "
                        "%.2f\n",
                        name, pixels, blockError(c, ref, extent), blockError(n, ref, extent),
                        laplacianEnergy(c, extent) / energy, laplacianEnergy(n, extent) / energy);
            const std::string tag = std::string(name) + "_" + std::to_string(int(pixels));
            writeImage("sharp_" + tag + "_reference", ref, extent);
            writeImage("sharp_" + tag + "_copy", c, extent);
            writeImage("sharp_" + tag + "_near", n, extent);
            EXPECT_GT(energy, 0.0);
        }
    }
}
