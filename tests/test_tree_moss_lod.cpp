// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - fullDetailMatchesTheCpu: seen from close by, every copy writes its full patch, where scatterMossPatches()
 *   puts it
 * - fartherCopiesUseCoarserLevels: from farther away the copies write fewer Gaussians, a quarter per level
 * - crossfadeSplitsTheOpacity: within a band around a boundary a copy writes both levels, their opacities scaled
 *   so that they add up to one level's
 * - copiesOutsideTheViewAreCulled: looking away, no copy writes anything and the pool stays transparent
 * - aFullPoolLeavesCopiesOut: copies that do not fit the pool are left out and counted
 * - coarserLevelsKeepTheCoverage: rendered from a distance, the moss covers nearly as many pixels as at full detail
 **/

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>

#include "klartraum/gaussian_splatting_factory.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/tree/tree_moss_lod.hpp"

#include "tree_test_helpers.hpp"

using namespace klartraum;
using namespace klartraum::tree_test;

namespace {

// A square patch of 40 x 40 flat green Gaussians, 0.1 wide.
MossPatch squarePatch() {
    MossPatch patch;
    for (int i = 0; i < 40; ++i) {
        for (int j = 0; j < 40; ++j) {
            Gaussian3D g{};
            g.position = {-0.05f + 0.0025f * float(i), 0.001f, -0.05f + 0.0025f * float(j)};
            g.rotation = {0.0f, 0.0f, 0.0f, 1.0f};
            g.scale = {0.0015f, 0.0002f, 0.0015f};
            g.color = {-0.8f, 0.6f, -0.8f};
            g.alpha = 0.9f;
            patch.push_back(g);
        }
    }
    return patch;
}

// Copies of the patch on a flat ground at y = 0, in a row along x.
std::vector<TreeLodInstance> row(int count) {
    std::vector<TreeLodInstance> copies;
    for (int k = 0; k < count; ++k) {
        TreeLodInstance inst{};
        inst.posYaw = glm::vec4(-0.3f + 0.15f * float(k), 0.0f, 0.0f, 0.3f * float(k));
        inst.plane = glm::vec4(0.0f, 0.0f, 1.0f, 0.0f);
        copies.push_back(inst);
    }
    return copies;
}

struct MossScene {
    MossScene(VulkanContext& vc, int copies, uint32_t pool, glm::vec4 boundaries = glm::vec4(160, 48, 12, 0.15f),
              VkExtent2D extent = {256, 256})
        : vc(vc),
          extent(extent) {
        // A generator with a trunk only; the trees are not part of the Gaussians here.
        TreeTypeParams p = tree_presets::quakingAspen();
        p.ratios.x = 1.0f;
        p.leafCount.x = 0.0f;
        const std::vector<TreeTypeParams> types = {p};
        const std::vector<TreeInstance> trees = {makeInstance(0, 1, glm::vec3(10.0f, 0.0f, 0.0f), 0.01f)};
        generator = std::make_shared<TreeGenerator>(vc, types, trees,
                                                    TreeCapacity::fromReference(types, trees, defaultTreeFrame()));
        camera = std::make_shared<CameraUboType>();
        TreeRenderSettings values = defaultTreeRenderSettings(extent);
        settings = std::make_shared<HostValues<TreeRenderSettings>>(vc, std::vector<TreeRenderSettings>{values});
        TreeGaussianSources sources;
        sources.trees = false;
        sources.extraSlots = pool;
        gaussians = std::make_unique<TreeGaussians>(vc, *generator, camera, settings, std::move(sources));
        patches = {squarePatch()};
        instances = row(copies);
        moss = std::make_unique<TreeMossLod>(vc, *gaussians, patches, instances, camera, settings, BufferRef{},
                                             boundaries);
        target = std::make_shared<OffscreenTarget>(vc, extent, 1);
        splatting = createGaussianSplatting(vc, GsplatBackend::Raster, target, camera, moss->getGaussians());
        graph = std::make_unique<ComputeGraph>(vc, 1);
        graph->compileFrom(splatting);
        gaussians->initializeBuffers(1);
    }

    std::vector<uint8_t> render(glm::vec3 eye, glm::vec3 lookAt) {
        setCamera(camera->ubo, eye, lookAt, 40.0f, float(extent.width) / float(extent.height));
        camera->update(0);
        graph->submitAndWait(vc.getGraphicsQueue(), 0);
        return readImage(vc, target->getImage(0), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, extent);
    }

    template <typename T>
    std::vector<T> read(const BufferRef& ref) {
        auto buffer = std::dynamic_pointer_cast<BufferElement<VulkanBuffer<T>>>(ref.buffer());
        std::vector<T> values(buffer->getBuffer(0).getSize());
        buffer->getBuffer(0).memcopyTo(values);
        return values;
    }

    VulkanContext& vc;
    VkExtent2D extent;
    std::shared_ptr<TreeGenerator> generator;
    std::shared_ptr<CameraUboType> camera;
    std::shared_ptr<HostValues<TreeRenderSettings>> settings;
    std::unique_ptr<TreeGaussians> gaussians;
    std::vector<MossPatch> patches;
    std::vector<TreeLodInstance> instances;
    std::unique_ptr<TreeMossLod> moss;
    std::shared_ptr<OffscreenTarget> target;
    std::shared_ptr<ComputeGraphElement> splatting;
    std::unique_ptr<ComputeGraph> graph;
};

} // namespace

class TreeMossLodTest : public ::testing::Test {
protected:
    void SetUp() override {
        frontend = std::make_unique<HeadlessFrontend>();
        vc = &frontend->getKlartraumEngine().getVulkanContext();
    }
    void TearDown() override { frontend.reset(); }

    std::unique_ptr<HeadlessFrontend> frontend;
    VulkanContext* vc = nullptr;
};

// Boundaries of 0 pixels: every copy in view at full detail.
const glm::vec4 kFullDetail(0.0f, 0.0f, 0.0f, 0.0f);

TEST_F(TreeMossLodTest, fullDetailMatchesTheCpu) {
    MossScene scene(*vc, 5, 20000, kFullDetail);
    scene.render(glm::vec3(0.0f, 0.6f, 0.6f), glm::vec3(0.0f));
    const std::vector<uint32_t> counter = scene.moss->readCounter();
    EXPECT_EQ(counter[0], 5u * 1600u);
    EXPECT_EQ(counter[1], 0u);

    // The written Gaussians, sorted, against the CPU expansion, sorted.
    const uint32_t offset = scene.gaussians->getExtraOffset();
    const std::vector<glm::vec3> pos = scene.read<glm::vec3>(scene.moss->getGaussians().pos);
    std::vector<glm::vec3> gpu(pos.begin() + offset, pos.begin() + offset + counter[0]);
    std::vector<glm::vec3> cpu;
    for (const Gaussian3D& g : scatterMossPatches(scene.patches, scene.instances)) {
        cpu.emplace_back(g.position[0], g.position[1], g.position[2]);
    }
    auto less = [](const glm::vec3& a, const glm::vec3& b) {
        return std::tie(a.x, a.z, a.y) < std::tie(b.x, b.z, b.y);
    };
    std::sort(gpu.begin(), gpu.end(), less);
    std::sort(cpu.begin(), cpu.end(), less);
    ASSERT_EQ(gpu.size(), cpu.size());
    float worst = 0.0f;
    for (size_t i = 0; i < gpu.size(); ++i) {
        worst = std::max(worst, glm::length(gpu[i] - cpu[i]));
    }
    EXPECT_LT(worst, 1e-4f);
}

TEST_F(TreeMossLodTest, fartherCopiesUseCoarserLevels) {
    // A copy is 0.14 across (with its corners); level boundaries scaled to this test's distances.
    MossScene scene(*vc, 5, 20000, glm::vec4(160, 48, 12, 0.0f));
    uint32_t previous = UINT32_MAX;
    for (float distance : {0.3f, 1.0f, 3.0f, 9.0f}) {
        scene.render(glm::vec3(0.0f, distance, 0.01f), glm::vec3(0.0f));
        const uint32_t written = scene.moss->readCounter()[0];
        std::cout << "distance " << distance << ": " << written << " Gaussians" << std::endl;
        EXPECT_LT(written, previous) << distance;
        previous = written;
    }
    // Far away every copy is at level 3: 1600 / 64 = 25 Gaussians.
    EXPECT_EQ(previous, 5u * 25u);
}

TEST_F(TreeMossLodTest, crossfadeSplitsTheOpacity) {
    // One copy; boundary 1 placed at its current size, so it is half way between levels 0 and 1.
    MossScene probe(*vc, 1, 4000, glm::vec4(1e6f, 1e5f, 1e4f, 0.15f));
    const glm::vec3 eye(-0.3f, 1.0f, 0.01f), at(-0.3f, 0.0f, 0.0f);
    const float focal = std::abs(0.5f * 256.0f / std::tan(glm::radians(20.0f)));
    const float radius = std::sqrt(2.0f) * 0.05f;
    const float pixels = tree::treePixelDiameter(eye, at, radius, focal);
    MossScene scene(*vc, 1, 4000, glm::vec4(pixels, pixels / 10.0f, pixels / 100.0f, 0.15f));
    scene.render(eye, at);
    const uint32_t written = scene.moss->readCounter()[0];
    EXPECT_EQ(written, 1600u + 400u);
    const std::vector<glm::vec4> colAlpha = scene.read<glm::vec4>(scene.moss->getGaussians().colAlpha);
    const uint32_t offset = scene.gaussians->getExtraOffset();
    double fine = 0.0, coarse = 0.0;
    for (uint32_t k = 0; k < written; ++k) {
        (k < 1600u ? fine : coarse) += colAlpha[offset + k].w;
    }
    // Half of level 0's opacity and half of level 1's.
    EXPECT_NEAR(fine / (1600.0 * 0.9), 0.5, 0.01);
    EXPECT_NEAR(coarse / (400.0 * 0.9), 0.5, 0.01);
}

TEST_F(TreeMossLodTest, copiesOutsideTheViewAreCulled) {
    MossScene scene(*vc, 5, 20000);
    const auto image = scene.render(glm::vec3(0.0f, 0.4f, 0.3f), glm::vec3(0.0f, 0.8f, 2.0f));
    EXPECT_EQ(scene.moss->readCounter()[0], 0u);
    const std::vector<glm::vec4> colAlpha = scene.read<glm::vec4>(scene.moss->getGaussians().colAlpha);
    for (uint32_t k = scene.gaussians->getExtraOffset(); k < colAlpha.size(); ++k) {
        ASSERT_EQ(colAlpha[k].w, 0.0f);
    }
    EXPECT_EQ(coveredCount(image, scene.extent), 0u);
}

TEST_F(TreeMossLodTest, aFullPoolLeavesCopiesOut) {
    MossScene scene(*vc, 5, 3500, kFullDetail);
    scene.render(glm::vec3(0.0f, 0.6f, 0.6f), glm::vec3(0.0f));
    const std::vector<uint32_t> counter = scene.moss->readCounter();
    EXPECT_EQ(counter[1], 3u);    // two copies of 1600 fit
    EXPECT_GT(counter[0], 3500u); // reserved, though not all written
}

TEST_F(TreeMossLodTest, coarserLevelsKeepTheCoverage) {
    const glm::vec3 eye(0.0f, 2.0f, 0.01f);
    MossScene full(*vc, 5, 20000, kFullDetail);
    // Pixels the moss covers solidly, and its summed brightness.
    auto measure = [](const std::vector<uint8_t>& image, VkExtent2D extent) {
        size_t solid = 0;
        double brightness = 0.0;
        for (size_t i = 0; i < image.size(); i += 4) {
            const int sum = image[i] + image[i + 1] + image[i + 2];
            solid += sum > 120 ? 1 : 0;
            brightness += sum;
        }
        return std::make_pair(solid, brightness);
    };
    const auto fullImage = full.render(eye, glm::vec3(0.0f));
    writePpm("moss_lod_full", fullImage, full.extent);
    const auto fullMeasure = measure(fullImage, full.extent);
    MossScene lod(*vc, 5, 20000);
    const auto image = lod.render(eye, glm::vec3(0.0f));
    writePpm("moss_lod_coverage", image, lod.extent);
    const auto lodMeasure = measure(image, lod.extent);
    std::cout << "solid pixels: full " << fullMeasure.first << ", level of detail " << lodMeasure.first
              << "; brightness ratio " << lodMeasure.second / fullMeasure.second << ", with "
              << lod.moss->readCounter()[0] << " of " << 5 * 1600 << " Gaussians" << std::endl;
    EXPECT_LT(lod.moss->readCounter()[0], 5u * 1600u / 4u);
    EXPECT_NEAR(double(lodMeasure.first) / double(fullMeasure.first), 1.0, 0.1);
    EXPECT_NEAR(lodMeasure.second / fullMeasure.second, 1.0, 0.1);
}
