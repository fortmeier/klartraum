// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - exactCountSampling: the title sampled for a given number of splats gives exactly that many,
 *   all inside the letters, for counts below and above the natural count of the spacing
 * - placementFacesCamera: text placed for a camera is centred on the pivot, faces the camera and
 *   reads left to right, bottom to top in the camera's right and up directions
 * - placedTitleRendersLikeMask: the title placed for the lantern camera (from above and the
 *   side) and rendered matches the FreeType mask (IoU >= 0.90); frame in
 *   build/TestingOutput/gaussian_morph/
 * - matchingIsPermutation: the screen-space matching uses every text splat exactly once
 * - matchingFindsIdenticalPoints: two copies of a point set in shuffled order are matched point
 *   to point
 * - matchingIsCoherent: for the title and the lantern, neighbouring text splats get partners in
 *   the scene that are close to each other (mean distance < 0.3x that of a random matching)
 * - prepareTextMorphOrdersLikeScene: the prepared text is the placed text in matched order, with
 *   wave delays from 0 (leftmost) to 1 (rightmost)
 * - timelineHoldsMorphsAndSpins: the default shot has 255 frames at 30 fps; the first 60 hold
 *   the title with the camera still, the morph ends at 3.5 s, the camera speeds up without a jerk
 *   and turns once by the end
 * - morphEndpointsAreExact: for every variant, progress 0 gives the start set and progress 1 the
 *   end set, bit for bit
 * - morphIsContinuous: for every variant, in 45 steps no Gaussian makes a jump (no step longer
 *   than 15 % of its path, scaled up for the shorter motion of each Gaussian in the waves), also
 *   with a fractional number of swirl turns, and opacity and size stay between their start and
 *   end values
 * - waveStartsInOrder: in the wave and the swirl-wave, Gaussians whose delay has not come yet are still at their
 *   start, earlier ones have moved
 * - swirlCirclesTheAxis: halfway through the swirl, each Gaussian's distance from the axis and
 *   height along it are halfway between start and end, while it has turned around the axis
 * - scatterLeavesTheLine: halfway through the scatter, every Gaussian is between a quarter and
 *   half the scatter radius away from the midpoint of its straight path
 * - swirlScatterTurnsTheScatter: the swirl-scatter is the scatter turned around the axis: same
 *   distance from the axis and height, different place
 * - explosionsBurstOutward: at the end of the burst, every Gaussian of the explode variants is
 *   between half and the full explode radius from its start, moved away from the centre, and in
 *   the shatter towards the given direction
 * - morphRendersTitleThenLantern: the morph of the title into the lantern, rendered at progress
 *   0, looks like the placed title and at progress 1 like the lantern rendered directly
 *
 * The tests that set text (exactCountSampling, placedTitleRendersLikeMask, matchingIsCoherent,
 * morphRendersTitleThenLantern) are built with KLARTRAUM_WITH_FREETYPE only.
 **/

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <glm/gtc/matrix_transform.hpp>

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/gaussianmorph.hpp"
#include "klartraum/computegraph/hostvalues.hpp"
#include "klartraum/gaussian_data_standard.hpp"
#include "klartraum/gaussian_morph.hpp"
#include "klartraum/gaussian_splatting_factory.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/interface_camera_orbit.hpp"
#include "klartraum/offscreen_target.hpp"
#include "klartraum/text_gaussians.hpp"

#include "test_scene.hpp"

using namespace klartraum;

namespace {

#ifdef KLARTRAUM_WITH_FREETYPE
std::shared_ptr<Font> loadFont(const std::string& name) {
    try {
        return std::make_shared<Font>("fonts/" + name + ".ttf");
    } catch (const std::exception&) {
        return nullptr;
    }
}

#define REQUIRE_FONT(var, name)                                                                                        \
    auto var = loadFont(name);                                                                                         \
    if (!var) {                                                                                                        \
        GTEST_SKIP() << "font not available: " << name << " (configure with KLARTRAUM_DOWNLOAD_FONTS=ON)";             \
    }

// The promo title with the wordmark `size` em high.
TitleLayout promoTitle(const std::shared_ptr<Font>& font, const std::shared_ptr<Font>& mono, float size) {
    TitleLayout layout;
    layout.lines.push_back({"klartraum", font, 500.0f, size, 0.0f});
    layout.lines.push_back({"tech demo 000", mono, 400.0f, 0.4f * size, 0.05f});
    layout.pixelsPerUnit = 1024.0f / size;
    return layout;
}

// Reads an OffscreenTarget image back as tightly-packed BGRA bytes; the
// backends leave it in TRANSFER_SRC_OPTIMAL.
std::vector<uint8_t> readImage(VulkanContext& vc, VkImage image, VkExtent2D extent) {
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
#endif // KLARTRAUM_WITH_FREETYPE

std::vector<Gaussian3D> randomPoints(size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<Gaussian3D> points(n);
    for (auto& g : points) {
        g.position = {u(rng), u(rng), u(rng)};
        g.rotation = {0.0f, 0.0f, 0.0f, 1.0f};
    }
    return points;
}

// Random Gaussians with every attribute set: unit rotations, positive sizes,
// opacities in (0, 1) and SH coefficients.
std::vector<Gaussian3D> randomGaussians(size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<Gaussian3D> gaussians(n);
    for (auto& g : gaussians) {
        g.position = {u(rng), u(rng), u(rng)};
        const glm::quat q = glm::normalize(glm::quat(u(rng), u(rng), u(rng), u(rng)));
        g.rotation = {q.x, q.y, q.z, q.w};
        g.scale = {0.01f + 0.05f * (u(rng) + 1.0f), 0.01f + 0.05f * (u(rng) + 1.0f), 0.01f + 0.05f * (u(rng) + 1.0f)};
        g.color = {u(rng), u(rng), u(rng)};
        g.alpha = 0.05f + 0.45f * (u(rng) + 1.0f);
        for (int k = 0; k < 15; ++k) {
            g.shR[k] = u(rng);
            g.shG[k] = u(rng);
            g.shB[k] = u(rng);
        }
    }
    return gaussians;
}

template <typename T>
std::vector<T> readBack(const BufferRef& ref) {
    auto element = std::dynamic_pointer_cast<TemplatedBufferElementInterface<VulkanBuffer<T>>>(ref.buffer());
    if (!element) {
        throw std::runtime_error("unexpected buffer type");
    }
    std::vector<T> data(element->getBuffer(0).getBufferMemSize() / sizeof(T));
    element->getBuffer(0).memcopyTo(data);
    return data;
}

std::vector<Gaussian3D> readBack(const GaussianSoABuffers& buffers) {
    const auto pos = readBack<glm::vec3>(buffers.pos);
    const auto rot = readBack<glm::vec4>(buffers.rot);
    const auto scale = readBack<glm::vec3>(buffers.scale);
    const auto colAlpha = readBack<glm::vec4>(buffers.colAlpha);
    const auto shR = readBack<float>(buffers.shR);
    const auto shG = readBack<float>(buffers.shG);
    const auto shB = readBack<float>(buffers.shB);
    const uint32_t n = buffers.count;
    std::vector<Gaussian3D> result(n);
    for (uint32_t i = 0; i < n; ++i) {
        auto& g = result[i];
        g.position = {pos[i].x, pos[i].y, pos[i].z};
        g.rotation = {rot[i].x, rot[i].y, rot[i].z, rot[i].w};
        g.scale = {scale[i].x, scale[i].y, scale[i].z};
        g.color = {colAlpha[i].x, colAlpha[i].y, colAlpha[i].z};
        g.alpha = colAlpha[i].w;
        for (uint32_t k = 0; k < 15; ++k) {
            g.shR[k] = shR[k * n + i];
            g.shG[k] = shG[k * n + i];
            g.shB[k] = shB[k * n + i];
        }
    }
    return result;
}

bool identical(const Gaussian3D& a, const Gaussian3D& b) {
    return a.position == b.position && a.rotation == b.rotation && a.scale == b.scale && a.color == b.color &&
           a.alpha == b.alpha && a.shR == b.shR && a.shG == b.shG && a.shB == b.shB;
}

const std::vector<MorphVariant> kVariants = {MorphVariant::Straight, MorphVariant::Swirl, MorphVariant::Scatter,
                                             MorphVariant::Wave};

// A compiled graph that morphs `start` into `end`; run() evaluates it for
// the given parameters and reads the result back.
class MorphRunner {
public:
    MorphRunner(VulkanContext& vc, const std::vector<Gaussian3D>& start, const std::vector<Gaussian3D>& end,
                const std::vector<float>& delays)
        : vc(vc),
          startData(std::make_shared<GaussianDataStandard>(vc, start)),
          endData(std::make_shared<GaussianDataStandard>(vc, end)),
          params(std::make_shared<HostFloat>(vc, MorphParams{}.values())) {
        morph = createGaussianMorph(vc, startData->buffers(), endData->buffers(), BufferRef{params},
                                    uploadFloats(vc, delays, "MorphDelays"));
        graph = std::make_unique<ComputeGraph>(vc, 1);
        graph->compileFrom(morph.element);
    }

    std::vector<Gaussian3D> run(const MorphParams& values) {
        params->set(values.values());
        graph->submitAndWait(vc.getGraphicsQueue(), 0);
        return readBack(morph.output);
    }

private:
    VulkanContext& vc;
    std::shared_ptr<GaussianDataStandard> startData;
    std::shared_ptr<GaussianDataStandard> endData;
    std::shared_ptr<HostFloat> params;
    GaussianMorphResult morph;
    std::unique_ptr<ComputeGraph> graph;
};

class GaussianMorphGpu : public ::testing::Test {
protected:
    void SetUp() override { frontend = std::make_unique<HeadlessFrontend>(); }
    void TearDown() override { frontend.reset(); }
    VulkanContext& vc() { return frontend->getKlartraumEngine().getVulkanContext(); }

    std::unique_ptr<HeadlessFrontend> frontend;
};

float easeInOutCubic(float s) { return s < 0.5f ? 4.0f * s * s * s : 1.0f - std::pow(-2.0f * s + 2.0f, 3.0f) / 2.0f; }

glm::vec3 positionOf(const Gaussian3D& g) { return {g.position[0], g.position[1], g.position[2]}; }

#ifdef KLARTRAUM_WITH_FREETYPE
// The lantern camera of test_scene::frameLantern() without a Vulkan context.
glm::mat4 lanternView(const test_scene::Scene& scene) {
    const glm::mat4 view = glm::lookAt(glm::vec3(0.35f, 0.0f, 0.0f), glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f)) *
                           glm::rotate(glm::mat4(1.0f), -0.5f, glm::vec3(0.0f, 0.0f, 1.0f)) *
                           glm::rotate(glm::mat4(1.0f), 0.9f, glm::vec3(0.0f, 1.0f, 0.0f)) *
                           glm::translate(glm::mat4(1.0f), -scene.centre);
    return view;
}

void writePPM(const std::filesystem::path& path, const std::vector<uint8_t>& bgra, VkExtent2D extent) {
    std::ofstream out(path, std::ios::binary);
    out << "P6\n" << extent.width << " " << extent.height << "\n255\n";
    for (size_t i = 0; i < bgra.size(); i += 4) {
        const char rgb[3] = {char(bgra[i + 2]), char(bgra[i + 1]), char(bgra[i])};
        out.write(rgb, 3);
    }
}

#endif // KLARTRAUM_WITH_FREETYPE

} // namespace

#ifdef KLARTRAUM_WITH_FREETYPE
TEST(GaussianMorph, exactCountSampling) {
    REQUIRE_FONT(font, "Unbounded");
    REQUIRE_FONT(mono, "JetBrainsMono");
    const CoverageMask mask = rasterizeTitle(promoTitle(font, mono, 0.5f));
    TextSplatParams params;
    params.spacing = 0.01f;
    const size_t natural = sampleTextGaussians(mask, params).size();
    ASSERT_GT(natural, 1000u);

    for (uint32_t count : {uint32_t(natural / 3), uint32_t(natural * 3 + 17)}) {
        SCOPED_TRACE(count);
        const auto splats = sampleTextGaussians(mask, params, count);
        ASSERT_EQ(splats.size(), count);
        size_t outside = 0;
        for (const Gaussian3D& g : splats) {
            outside += mask.sample(g.position[0], g.position[1]) < params.threshold;
        }
        EXPECT_EQ(outside, 0u);
    }
}
#endif // KLARTRAUM_WITH_FREETYPE

TEST(GaussianMorph, placementFacesCamera) {
    // A camera above and to the side of the pivot, two units away.
    const glm::vec3 centre(0.2f, -0.1f, 0.3f);
    const glm::vec3 eye = centre + 2.0f * glm::normalize(glm::vec3(1.0f, 0.8f, 1.5f));
    CameraMVP mvp;
    mvp.view = glm::lookAt(eye, centre, glm::vec3(0.0f, 1.0f, 0.0f));

    // Three splats: at the text origin, one unit right and one unit up.
    std::vector<Gaussian3D> splats(3);
    for (auto& g : splats) {
        g.rotation = {0.0f, 0.0f, 0.0f, 1.0f};
        g.scale = {0.1f, 0.1f, 0.01f};
    }
    splats[1].position = {1.0f, 0.0f, 0.0f};
    splats[2].position = {0.0f, 1.0f, 0.0f};
    placeFacingCamera(splats, mvp.view, centre);

    const glm::vec3 origin(splats[0].position[0], splats[0].position[1], splats[0].position[2]);
    EXPECT_NEAR(glm::distance(origin, centre), 0.0f, 1e-5f);

    // In view space: right is +x, up is +y, the camera looks down -z.
    auto toView = [&](const Gaussian3D& g) {
        return glm::vec3(mvp.view * glm::vec4(g.position[0], g.position[1], g.position[2], 1.0f));
    };
    const glm::vec3 o = toView(splats[0]);
    const glm::vec3 right = toView(splats[1]) - o;
    const glm::vec3 up = toView(splats[2]) - o;
    EXPECT_NEAR(right.x, 1.0f, 1e-5f);
    EXPECT_NEAR(right.y, 0.0f, 1e-5f);
    EXPECT_NEAR(right.z, 0.0f, 1e-5f);
    EXPECT_NEAR(up.x, 0.0f, 1e-5f);
    EXPECT_NEAR(up.y, 1.0f, 1e-5f);
    EXPECT_NEAR(up.z, 0.0f, 1e-5f);
    EXPECT_NEAR(o.z, -2.0f, 1e-5f);

    // The flat axis of a splat points at the camera.
    const glm::quat q(splats[0].rotation[3], splats[0].rotation[0], splats[0].rotation[1], splats[0].rotation[2]);
    const glm::vec3 normal = glm::mat3(mvp.view) * (q * glm::vec3(0.0f, 0.0f, 1.0f));
    EXPECT_NEAR(normal.z, 1.0f, 1e-5f);
}

#ifdef KLARTRAUM_WITH_FREETYPE
TEST(GaussianMorph, placedTitleRendersLikeMask) {
    REQUIRE_FONT(font, "Unbounded");
    REQUIRE_FONT(mono, "JetBrainsMono");
    if (!std::filesystem::exists(test_scene::kLanternPath)) {
        GTEST_SKIP() << "SPZ scene not found: " << test_scene::kLanternPath;
    }
    const auto scene = test_scene::loadLantern();
    const VkExtent2D extent{1920, 1080};
    const float aspect = float(extent.width) / float(extent.height);

    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    auto target = std::make_shared<OffscreenTarget>(vc, extent, 1u);
    auto cameraUBO = std::make_shared<CameraUboType>();
    InterfaceCameraOrbit orbit(InterfaceCameraOrbit::UpDirection::Y);
    orbit.initialize(vc);
    test_scene::frameLantern(orbit, scene);
    orbit.setProjectionAspectRatio(aspect);
    CameraMVP mvp;
    orbit.update(mvp);

    // The wordmark spans 70 % of the image width at the pivot's distance.
    const double distance = 0.35;
    const double focal = 0.5 * extent.height / std::tan(0.5 * glm::radians(45.0));
    const float visibleWidth = float(extent.width * distance / focal);
    const float size = 0.7f * visibleWidth / measureLine({"klartraum", font, 500.0f, 1.0f, 0.0f});
    const CoverageMask mask = rasterizeTitle(promoTitle(font, mono, size));
    auto splats = sampleTextGaussians(mask, TextSplatParams{}, uint32_t(scene.gaussians.size()));
    placeFacingCamera(splats, mvp.view, scene.centre);

    auto model = std::make_shared<GaussianDataStandard>(vc, splats);
    auto splatting = createGaussianSplatting(vc, GsplatBackend::Compute, target, cameraUBO, model);
    std::vector<uint8_t> pixels;
    {
        ComputeGraph graph(vc, 1);
        graph.compileFrom(splatting);
        orbit.update(cameraUBO->ubo);
        cameraUBO->update(0);
        graph.submitAndWait(vc.getGraphicsQueue(), 0);
        pixels = readImage(vc, target->getImage(0), extent);
    }
    splatting.reset();
    target.reset();
    const std::filesystem::path outDir = "build/TestingOutput/gaussian_morph";
    std::filesystem::create_directories(outDir);
    writePPM(outDir / "placed_title.ppm", pixels, extent);

    // The text faces the camera at the pivot's distance, so each pixel
    // centre maps onto the text plane by the pinhole model.
    size_t both = 0, either = 0;
    for (uint32_t y = 0; y < extent.height; ++y) {
        for (uint32_t x = 0; x < extent.width; ++x) {
            const float tx = float((x + 0.5 - 0.5 * extent.width) * distance / focal);
            const float ty = float((0.5 * extent.height - (y + 0.5)) * distance / focal);
            const bool expected = mask.sample(tx, ty) >= 0.5f;
            const size_t i = (size_t(y) * extent.width + x) * 4;
            const bool rendered = (int(pixels[i]) + pixels[i + 1] + pixels[i + 2]) / 3 >= 128;
            both += expected && rendered;
            either += expected || rendered;
        }
    }
    ASSERT_GT(either, 0u);
    const double iou = double(both) / double(either);
    std::cout << "placed title: " << splats.size() << " splats, IoU " << iou << std::endl;
    EXPECT_GE(iou, 0.90);
}
#endif // KLARTRAUM_WITH_FREETYPE

TEST(GaussianMorph, matchingIsPermutation) {
    const auto text = randomPoints(1001, 1);
    const auto scene = randomPoints(1001, 2);
    const glm::mat4 view = glm::lookAt(glm::vec3(0.0f, 0.0f, 3.0f), glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    const std::vector<uint32_t> match = matchByScreenPosition(text, scene, view);
    ASSERT_EQ(match.size(), scene.size());
    std::vector<int> used(text.size(), 0);
    for (uint32_t m : match) {
        ASSERT_LT(m, text.size());
        ++used[m];
    }
    EXPECT_EQ(std::count(used.begin(), used.end(), 1), long(text.size()));
}

TEST(GaussianMorph, matchingFindsIdenticalPoints) {
    const auto scene = randomPoints(2000, 3);
    std::vector<uint32_t> order(scene.size());
    std::iota(order.begin(), order.end(), 0u);
    std::shuffle(order.begin(), order.end(), std::mt19937(4));
    std::vector<Gaussian3D> text(scene.size());
    for (size_t i = 0; i < order.size(); ++i) {
        text[i] = scene[order[i]];
    }
    const glm::mat4 view = glm::lookAt(glm::vec3(0.0f, 0.0f, 3.0f), glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    const std::vector<uint32_t> match = matchByScreenPosition(text, scene, view);
    for (size_t i = 0; i < scene.size(); ++i) {
        EXPECT_EQ(positionOf(text[match[i]]), positionOf(scene[i])) << i;
    }
}

#ifdef KLARTRAUM_WITH_FREETYPE
TEST(GaussianMorph, matchingIsCoherent) {
    REQUIRE_FONT(font, "Unbounded");
    REQUIRE_FONT(mono, "JetBrainsMono");
    if (!std::filesystem::exists(test_scene::kLanternPath)) {
        GTEST_SKIP() << "SPZ scene not found: " << test_scene::kLanternPath;
    }
    const auto scene = test_scene::loadLantern();
    const glm::mat4 view = lanternView(scene);
    const CoverageMask mask = rasterizeTitle(promoTitle(font, mono, 0.05f));
    auto text = sampleTextGaussians(mask, TextSplatParams{}, uint32_t(scene.gaussians.size()));
    placeFacingCamera(text, view, scene.centre);

    const std::vector<uint32_t> match = matchByScreenPosition(text, scene.gaussians, view);
    std::vector<uint32_t> shuffled(match);
    std::shuffle(shuffled.begin(), shuffled.end(), std::mt19937(5));
    // Where each text splat ends up, for the matching and for a random one.
    std::vector<uint32_t> target(text.size()), randomTarget(text.size());
    for (uint32_t i = 0; i < match.size(); ++i) {
        target[match[i]] = i;
        randomTarget[shuffled[i]] = i;
    }
    // Text splats come in grid order, so consecutive ones are mostly
    // neighbours on the letters; their partners should be close as well.
    double matched = 0.0, random = 0.0;
    for (size_t i = 0; i + 1 < text.size(); ++i) {
        matched += glm::distance(positionOf(scene.gaussians[target[i]]), positionOf(scene.gaussians[target[i + 1]]));
        random += glm::distance(positionOf(scene.gaussians[randomTarget[i]]),
                                positionOf(scene.gaussians[randomTarget[i + 1]]));
    }
    std::cout << "mean distance between the partners of neighbours: matched " << matched / text.size() << ", random "
              << random / text.size() << std::endl;
    // Measured: 0.038 against 0.195 for a random matching.
    EXPECT_LT(matched, 0.3 * random);
}
#endif // KLARTRAUM_WITH_FREETYPE

TEST(GaussianMorph, prepareTextMorphOrdersLikeScene) {
    auto text = randomPoints(500, 6);
    for (auto& g : text) {
        g.position[2] = 0.0f;
    }
    const auto scene = randomPoints(500, 7);
    const glm::vec3 centre(0.1f, 0.2f, 0.3f);
    const glm::mat4 view = glm::lookAt(centre + glm::vec3(1.0f, 0.5f, 2.0f), centre, glm::vec3(0.0f, 1.0f, 0.0f));
    const TextMorphStart start = prepareTextMorph(text, scene, view, centre);

    auto placed = text;
    placeFacingCamera(placed, view, centre);
    const std::vector<uint32_t> match = matchByScreenPosition(placed, scene, view);
    ASSERT_EQ(start.gaussians.size(), scene.size());
    ASSERT_EQ(start.delays.size(), scene.size());
    for (size_t i = 0; i < scene.size(); ++i) {
        EXPECT_TRUE(identical(start.gaussians[i], placed[match[i]])) << i;
        const float x = text[match[i]].position[0];
        EXPECT_GE(start.delays[i], 0.0f);
        EXPECT_LE(start.delays[i], 1.0f);
        if (x == std::min_element(text.begin(), text.end(), [](const auto& a, const auto& b) {
                     return a.position[0] < b.position[0];
                 })->position[0]) {
            EXPECT_EQ(start.delays[i], 0.0f);
        }
    }
    EXPECT_EQ(*std::max_element(start.delays.begin(), start.delays.end()), 1.0f);
}

TEST(GaussianMorph, timelineHoldsMorphsAndSpins) {
    const MorphTimeline timeline;
    EXPECT_EQ(timeline.frameCount(), 255u);
    EXPECT_EQ(timeline.titleFrames(), 60u);
    for (uint32_t f = 0; f < 60; ++f) {
        EXPECT_EQ(timeline.progressAt(f), 0.0f);
        EXPECT_EQ(timeline.angleAt(f), 0.0);
    }
    EXPECT_GT(timeline.progressAt(61), 0.0f);
    EXPECT_LT(timeline.progressAt(104), 1.0f);
    EXPECT_EQ(timeline.progressAt(105), 1.0f);
    EXPECT_EQ(timeline.progressAt(254), 1.0f);

    // The angle grows every frame after the title, and the step per frame
    // never jumps: it grows evenly during the morph and then stays constant.
    double previousStep = 0.0;
    for (uint32_t f = 61; f < timeline.frameCount(); ++f) {
        const double step = timeline.angleAt(f) - timeline.angleAt(f - 1);
        EXPECT_GT(step, 0.0) << f;
        EXPECT_GE(step, previousStep - 1e-9) << f;
        EXPECT_LT(step - previousStep, 0.01) << f;
        previousStep = step;
    }
    const double rate = timeline.totalAngle() / (0.5 * 1.5 + 5.0);
    EXPECT_NEAR(previousStep, rate / 30.0, 1e-9);
    // One frame before the end of the shot, one frame's turn is left.
    EXPECT_NEAR(timeline.angleAt(254) + rate / 30.0, timeline.totalAngle(), 1e-9);
}

TEST_F(GaussianMorphGpu, morphEndpointsAreExact) {
    const auto start = randomGaussians(1000, 10);
    const auto end = randomGaussians(1000, 11);
    std::vector<float> delays(1000);
    for (size_t i = 0; i < delays.size(); ++i) {
        delays[i] = float(i) / 999.0f;
    }
    MorphRunner runner(vc(), start, end, delays);
    for (MorphVariant variant : kVariants) {
        SCOPED_TRACE(int(variant));
        MorphParams params;
        params.variant = variant;
        params.progress = 0.0f;
        auto result = runner.run(params);
        size_t wrong = 0;
        for (size_t i = 0; i < start.size(); ++i) {
            wrong += !identical(result[i], start[i]);
        }
        EXPECT_EQ(wrong, 0u) << "at progress 0";
        params.progress = 1.0f;
        result = runner.run(params);
        wrong = 0;
        for (size_t i = 0; i < end.size(); ++i) {
            wrong += !identical(result[i], end[i]);
        }
        EXPECT_EQ(wrong, 0u) << "at progress 1";
    }
}

TEST_F(GaussianMorphGpu, morphIsContinuous) {
    const size_t n = 500;
    const auto start = randomGaussians(n, 12);
    const auto end = randomGaussians(n, 13);
    std::vector<float> delays(n);
    for (size_t i = 0; i < n; ++i) {
        delays[i] = float(i) / float(n - 1);
    }
    MorphRunner runner(vc(), start, end, delays);
    const int steps = 45;
    for (MorphVariant variant : kVariants) {
        SCOPED_TRACE(int(variant));
        MorphParams params;
        params.variant = variant;
        // Swirls make whole turns; half a turn more must not show as a jump.
        params.swirlTurns = 1.5f;
        std::vector<float> longestStep(n, 0.0f), pathLength(n, 0.0f);
        std::vector<Gaussian3D> previous = start;
        size_t outOfRange = 0;
        for (int step = 1; step <= steps; ++step) {
            params.progress = float(step) / float(steps);
            const auto current = runner.run(params);
            for (size_t i = 0; i < n; ++i) {
                const float d = glm::distance(positionOf(current[i]), positionOf(previous[i]));
                longestStep[i] = std::max(longestStep[i], d);
                pathLength[i] += d;
                const float lo = std::min(start[i].alpha, end[i].alpha), hi = std::max(start[i].alpha, end[i].alpha);
                outOfRange += current[i].alpha < lo - 1e-6f || current[i].alpha > hi + 1e-6f;
                for (int k = 0; k < 3; ++k) {
                    const float a = start[i].scale[k], b = end[i].scale[k];
                    outOfRange += current[i].scale[k] < std::min(a, b) * (1.0f - 1e-5f) ||
                                  current[i].scale[k] > std::max(a, b) * (1.0f + 1e-5f);
                }
            }
            previous = current;
        }
        EXPECT_EQ(outOfRange, 0u);
        // In the wave each Gaussian moves during only (1 - spread) of the
        // morph, so its steps are that much longer.
        const bool wave = variant == MorphVariant::Wave || variant == MorphVariant::SwirlWave;
        const float movingPart = wave ? 1.0f - params.waveSpread : 1.0f;
        size_t jumps = 0;
        for (size_t i = 0; i < n; ++i) {
            jumps += longestStep[i] > 0.15f / movingPart * pathLength[i];
        }
        EXPECT_EQ(jumps, 0u);
    }
}

TEST_F(GaussianMorphGpu, waveStartsInOrder) {
    const size_t n = 400;
    const auto start = randomGaussians(n, 14);
    const auto end = randomGaussians(n, 15);
    std::vector<float> delays(n);
    for (size_t i = 0; i < n; ++i) {
        delays[i] = float(i) / float(n - 1);
    }
    MorphRunner runner(vc(), start, end, delays);
    for (MorphVariant variant : {MorphVariant::Wave, MorphVariant::SwirlWave}) {
        SCOPED_TRACE(int(variant));
        MorphParams params;
        params.variant = variant;
        params.waveSpread = 0.6f;
        params.progress = 0.3f;
        const auto result = runner.run(params);
        for (size_t i = 0; i < n; ++i) {
            if (delays[i] * params.waveSpread >= params.progress) {
                EXPECT_TRUE(identical(result[i], start[i])) << i;
            } else if (delays[i] * params.waveSpread < 0.5f * params.progress) {
                EXPECT_GT(glm::distance(positionOf(result[i]), positionOf(start[i])), 0.0f) << i;
            }
        }
    }
}

TEST_F(GaussianMorphGpu, swirlCirclesTheAxis) {
    const size_t n = 500;
    const auto start = randomGaussians(n, 16);
    const auto end = randomGaussians(n, 17);
    MorphRunner runner(vc(), start, end, std::vector<float>(n, 0.0f));
    MorphParams params;
    params.variant = MorphVariant::Swirl;
    params.centre = glm::vec3(0.1f, 0.0f, -0.2f);
    params.axis = glm::vec3(0.0f, 1.0f, 0.0f);
    params.progress = 0.5f;
    const auto swirl = runner.run(params);
    params.variant = MorphVariant::Straight;
    const auto straight = runner.run(params);

    const float s = easeInOutCubic(0.5f);
    size_t turned = 0;
    for (size_t i = 0; i < n; ++i) {
        auto cylinder = [&](const Gaussian3D& g) {
            const glm::vec3 v = positionOf(g) - params.centre;
            return glm::vec2(glm::length(glm::vec2(v.x, v.z)), v.y);
        };
        const glm::vec2 a = cylinder(start[i]), b = cylinder(end[i]), c = cylinder(swirl[i]);
        EXPECT_NEAR(c.x, a.x + s * (b.x - a.x), 1e-4f) << i;
        EXPECT_NEAR(c.y, a.y + s * (b.y - a.y), 1e-4f) << i;
        turned += glm::distance(positionOf(swirl[i]), positionOf(straight[i])) > 1e-3f;
    }
    EXPECT_GT(turned, n * 9 / 10);
}

TEST_F(GaussianMorphGpu, scatterLeavesTheLine) {
    const size_t n = 500;
    const auto start = randomGaussians(n, 18);
    const auto end = randomGaussians(n, 19);
    MorphRunner runner(vc(), start, end, std::vector<float>(n, 0.0f));
    MorphParams params;
    params.variant = MorphVariant::Scatter;
    params.scatterRadius = 0.4f;
    params.progress = 0.5f;
    const auto result = runner.run(params);
    for (size_t i = 0; i < n; ++i) {
        const glm::vec3 middle = 0.5f * (positionOf(start[i]) + positionOf(end[i]));
        const float offset = glm::distance(positionOf(result[i]), middle);
        EXPECT_GE(offset, 0.25f * params.scatterRadius - 1e-4f) << i;
        EXPECT_LE(offset, 0.5f * params.scatterRadius + 1e-4f) << i;
    }
}

TEST_F(GaussianMorphGpu, swirlScatterTurnsTheScatter) {
    const size_t n = 500;
    const auto start = randomGaussians(n, 22);
    const auto end = randomGaussians(n, 23);
    MorphRunner runner(vc(), start, end, std::vector<float>(n, 0.0f));
    MorphParams params;
    params.centre = glm::vec3(0.1f, 0.0f, -0.2f);
    params.axis = glm::vec3(0.0f, 1.0f, 0.0f);
    params.scatterRadius = 0.4f;
    params.progress = 0.4f;
    params.variant = MorphVariant::SwirlScatter;
    const auto swirled = runner.run(params);
    params.variant = MorphVariant::Scatter;
    const auto scattered = runner.run(params);

    // A turn around the axis keeps the distance from the axis and the height.
    auto cylinder = [&](const Gaussian3D& g) {
        const glm::vec3 v = positionOf(g) - params.centre;
        return glm::vec2(glm::length(glm::vec2(v.x, v.z)), v.y);
    };
    size_t turned = 0;
    for (size_t i = 0; i < n; ++i) {
        const glm::vec2 a = cylinder(swirled[i]), b = cylinder(scattered[i]);
        EXPECT_NEAR(a.x, b.x, 1e-4f) << i;
        EXPECT_NEAR(a.y, b.y, 1e-4f) << i;
        turned += glm::distance(positionOf(swirled[i]), positionOf(scattered[i])) > 1e-3f;
    }
    EXPECT_GT(turned, n * 9 / 10);
}

TEST_F(GaussianMorphGpu, explosionsBurstOutward) {
    const size_t n = 500;
    auto start = randomGaussians(n, 20);
    const auto end = randomGaussians(n, 21);
    MorphRunner runner(vc(), start, end, std::vector<float>(n, 0.0f));
    MorphParams params;
    params.centre = glm::vec3(0.0f);
    params.explodeRadius = 0.5f;
    params.explodeShare = 0.4f;
    params.direction = glm::normalize(glm::vec3(0.3f, 0.2f, 1.0f));
    params.progress = params.explodeShare;
    for (MorphVariant variant : {MorphVariant::Explode, MorphVariant::ExplodeSwirl, MorphVariant::Shatter}) {
        SCOPED_TRACE(int(variant));
        params.variant = variant;
        const auto result = runner.run(params);
        size_t wrong = 0;
        for (size_t i = 0; i < n; ++i) {
            const glm::vec3 a = positionOf(start[i]);
            const glm::vec3 moved = positionOf(result[i]) - a;
            const float distance = glm::length(moved);
            wrong += distance < 0.5f * params.explodeRadius - 1e-4f || distance > params.explodeRadius + 1e-4f;
            const glm::vec3 away = variant == MorphVariant::Shatter ? params.direction : a - params.centre;
            wrong += glm::dot(moved, away) <= 0.0f;
        }
        EXPECT_EQ(wrong, 0u);
    }
}

#ifdef KLARTRAUM_WITH_FREETYPE
TEST_F(GaussianMorphGpu, morphRendersTitleThenLantern) {
    REQUIRE_FONT(font, "Unbounded");
    REQUIRE_FONT(mono, "JetBrainsMono");
    if (!std::filesystem::exists(test_scene::kLanternPath)) {
        GTEST_SKIP() << "SPZ scene not found: " << test_scene::kLanternPath;
    }
    const auto scene = test_scene::loadLantern();
    const VkExtent2D extent{960, 540};
    auto target = std::make_shared<OffscreenTarget>(vc(), extent, 1u);
    auto cameraUBO = std::make_shared<CameraUboType>();
    InterfaceCameraOrbit orbit(InterfaceCameraOrbit::UpDirection::Y);
    orbit.initialize(vc());
    test_scene::frameLantern(orbit, scene);
    orbit.setProjectionAspectRatio(float(extent.width) / float(extent.height));
    CameraMVP mvp;
    orbit.update(mvp);

    const CoverageMask mask = rasterizeTitle(promoTitle(font, mono, 0.05f));
    const TextMorphStart start =
        prepareTextMorph(sampleTextGaussians(mask, TextSplatParams{}, uint32_t(scene.gaussians.size())),
                         scene.gaussians, mvp.view, scene.centre);

    // Renders `gaussians` with the same camera and target.
    auto render = [&](const GaussianSoABuffers& gaussians, const std::function<void()>& beforeRun) {
        auto splatting = createGaussianSplatting(vc(), GsplatBackend::Compute, target, cameraUBO, gaussians);
        ComputeGraph graph(vc(), 1);
        graph.compileFrom(splatting);
        orbit.update(cameraUBO->ubo);
        cameraUBO->update(0);
        beforeRun();
        graph.submitAndWait(vc().getGraphicsQueue(), 0);
        return readImage(vc(), target->getImage(0), extent);
    };
    auto meanAbsDifference = [](const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
        double sum = 0.0;
        for (size_t i = 0; i < a.size(); i += 4) {
            for (size_t c = 0; c < 3; ++c) {
                sum += std::abs(int(a[i + c]) - int(b[i + c]));
            }
        }
        return sum / double(a.size() / 4 * 3);
    };

    auto textData = std::make_shared<GaussianDataStandard>(vc(), start.gaussians);
    auto sceneData = std::make_shared<GaussianDataStandard>(vc(), scene.gaussians);
    const auto title = render(textData->buffers(), [] {});
    const auto lantern = render(sceneData->buffers(), [] {});

    auto params = std::make_shared<HostFloat>(vc(), MorphParams{}.values());
    const auto morph = createGaussianMorph(vc(), textData->buffers(), sceneData->buffers(), BufferRef{params},
                                           uploadFloats(vc(), start.delays, "MorphDelays"));
    const std::filesystem::path outDir = "build/TestingOutput/gaussian_morph";
    std::filesystem::create_directories(outDir);
    for (MorphVariant variant : kVariants) {
        SCOPED_TRACE(int(variant));
        MorphParams values;
        values.variant = variant;
        values.centre = scene.centre;
        values.scatterRadius = 0.2f;
        for (float progress : {0.0f, 0.5f, 1.0f}) {
            values.progress = progress;
            const auto frame = render(morph.output, [&] { params->set(values.values()); });
            const std::string name =
                "morph_" + std::to_string(int(variant)) + "_" + std::to_string(int(progress * 100)) + ".ppm";
            writePPM(outDir / name, frame, extent);
            if (progress == 0.0f) {
                EXPECT_EQ(meanAbsDifference(frame, title), 0.0);
            } else if (progress == 1.0f) {
                EXPECT_EQ(meanAbsDifference(frame, lantern), 0.0);
            } else {
                // Halfway, the frame is neither the title nor the lantern.
                EXPECT_GT(meanAbsDifference(frame, title), 1.0);
                EXPECT_GT(meanAbsDifference(frame, lantern), 1.0);
            }
        }
    }
}
#endif // KLARTRAUM_WITH_FREETYPE
