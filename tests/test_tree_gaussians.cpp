// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - unusedSlotsAreTransparent: the Gaussian count covers the capacities; slots of generated records are
 *   opaque (with no minimum stem radius), the others have opacity 0
 * - stemGaussiansLieOnTheTube: the Gaussians of a straight trunk sit just inside its surface, around and
 *   along each segment
 * - leafGaussiansLieOnTheLeaf: the Gaussians of a leaf on the z = 0 plane lie in the plane, inside the leaf
 * - silhouetteMatchesClassical: a preset tree splatted with either backend covers nearly the same pixels as
 *   its classical rendering
 **/

#include <gtest/gtest.h>

#include <cmath>
#include <memory>

#include "klartraum/gaussian_splatting_factory.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/tree/tree_gaussians.hpp"

#include "tree_test_helpers.hpp"

using namespace klartraum;
using namespace klartraum::tree;
using namespace klartraum::tree_test;

namespace {

TreeTypeParams straightTrunk() {
    TreeTypeParams p = tree_presets::quakingAspen();
    p.shapeBase = glm::vec4(3.0f, 0.0f, 2.0f, 0.0f);
    p.ratios = glm::vec4(1.0f, 0.05f, 1.0f, 0.0f);
    p.leafCount = glm::vec4(0.0f, 0.1f, 1.0f, 0.0f);
    p.nLength.x = 1.0f;
    p.nLengthV.x = 0.0f;
    p.nTaper.x = 0.0f;
    p.nCurveRes.x = 4.0f;
    p.nCurve.x = 0.0f;
    p.nCurveV.x = 0.0f;
    return p;
}

// Generator and TreeGaussians in a graph of their own, for reading the Gaussians back.
struct GaussianScene {
    GaussianScene(VulkanContext& vc, std::vector<TreeTypeParams> types, std::vector<TreeInstance> instances)
        : vc(vc) {
        const TreeFrame frame = defaultTreeFrame();
        generator = std::make_shared<TreeGenerator>(vc, types, instances,
                                                    TreeCapacity::fromReference(types, instances, frame), frame);
        // No stem is too thin to be drawn.
        TreeRenderSettings values = defaultTreeRenderSettings({256, 256});
        values.viewport.w = 0.0f;
        settings = std::make_shared<HostValues<TreeRenderSettings>>(vc, std::vector<TreeRenderSettings>{values});
        camera = std::make_shared<CameraUboType>();
        setCamera(camera->ubo, glm::vec3(0.0f, 1.0f, 0.5f), glm::vec3(0.0f, 1.0f, 0.0f), 40.0f, 1.0f);
        gaussians = std::make_unique<TreeGaussians>(vc, *generator, camera, settings);
        graph = std::make_unique<ComputeGraph>(vc, 1);
        graph->compileFrom(gaussians->getElement());
        camera->update(0);
        graph->submitAndWait(vc.getGraphicsQueue(), 0);
    }

    template <typename T>
    std::vector<T> read(const BufferRef& ref) {
        auto buffer = std::dynamic_pointer_cast<BufferElement<VulkanBuffer<T>>>(ref.buffer());
        std::vector<T> values(buffer->getBuffer(0).getSize());
        buffer->getBuffer(0).memcopyTo(values);
        return values;
    }

    VulkanContext& vc;
    std::shared_ptr<TreeGenerator> generator;
    std::shared_ptr<HostValues<TreeRenderSettings>> settings;
    std::shared_ptr<CameraUboType> camera;
    std::unique_ptr<TreeGaussians> gaussians;
    std::unique_ptr<ComputeGraph> graph;
};

} // namespace

class TreeGaussiansTest : public ::testing::Test {
protected:
    void SetUp() override {
        frontend = std::make_unique<HeadlessFrontend>();
        vc = &frontend->getKlartraumEngine().getVulkanContext();
    }
    void TearDown() override { frontend.reset(); }

    std::unique_ptr<HeadlessFrontend> frontend;
    VulkanContext* vc = nullptr;
};

TEST_F(TreeGaussiansTest, unusedSlotsAreTransparent) {
    GaussianScene scene(*vc, {tree_presets::sassafras()}, {makeInstance(0, 2, glm::vec3(0.0f), 0.1f)});
    const TreeCapacity& capacity = scene.generator->getCapacity();
    const GaussianSoABuffers& g = scene.gaussians->getGaussians();
    ASSERT_EQ(g.count,
              capacity.segments * TreeGaussians::kStemGaussians + capacity.leaves * TreeGaussians::kLeafGaussians);
    const std::vector<uint32_t> counters = scene.generator->readCounters();
    const std::vector<glm::vec4> colAlpha = scene.read<glm::vec4>(g.colAlpha);
    const uint32_t stemSlots = capacity.segments * TreeGaussians::kStemGaussians;
    for (uint32_t i = 0; i < g.count; ++i) {
        const bool used = i < stemSlots
                              ? i / TreeGaussians::kStemGaussians < counters[kTreeCounterSegments]
                              : (i - stemSlots) / TreeGaussians::kLeafGaussians < counters[kTreeCounterLeaves];
        if (used) {
            ASSERT_GT(colAlpha[i].w, 0.5f) << i;
        } else {
            ASSERT_EQ(colAlpha[i].w, 0.0f) << i;
        }
    }
}

TEST_F(TreeGaussiansTest, stemGaussiansLieOnTheTube) {
    GaussianScene scene(*vc, {straightTrunk()}, {makeInstance(0, 1)});
    const GaussianSoABuffers& g = scene.gaussians->getGaussians();
    const std::vector<glm::vec3> pos = scene.read<glm::vec3>(g.pos);
    // Trunk of length 2 and radius 0.1 along +y, in 4 segments.
    for (uint32_t i = 0; i < 4 * TreeGaussians::kStemGaussians; ++i) {
        const float distance = std::sqrt(pos[i].x * pos[i].x + pos[i].z * pos[i].z);
        EXPECT_NEAR(distance, 0.85f * 0.1f, 1e-4f) << i;
        EXPECT_GT(pos[i].y, 0.0f);
        EXPECT_LT(pos[i].y, 2.0f);
    }
    // Two rows per segment, at a quarter and three quarters of its length.
    EXPECT_NEAR(pos[0].y, 0.125f, 1e-4f);
    EXPECT_NEAR(pos[8].y, 0.375f, 1e-4f);
}

TEST_F(TreeGaussiansTest, leafGaussiansLieOnTheLeaf) {
    TreeTypeParams p = straightTrunk();
    p.ratios.y = 0.001f;
    p.leafCount = glm::vec4(1.0f, 0.8f, 0.8f, 0.0f);
    p.leafAngles = glm::vec4(0.0f);
    p.leafShape = glm::vec4(70.0f, 0.0f, 50.0f, 0.4f);
    p.leafLobes = glm::vec4(1.0f, 0.0f, 1.0f, 0.0f);
    GaussianScene scene(*vc, {p}, {makeInstance(0, 1)});
    const GaussianSoABuffers& g = scene.gaussians->getGaussians();
    const std::vector<glm::vec3> pos = scene.read<glm::vec3>(g.pos);
    const uint32_t first = scene.generator->getCapacity().segments * TreeGaussians::kStemGaussians;
    for (uint32_t i = first; i < first + TreeGaussians::kLeafGaussians; ++i) {
        // The leaf: base (0, 1, 0), 0.8 long up +y, 0.32 wide, on z = 0.
        EXPECT_NEAR(pos[i].z, 0.0f, 1e-5f) << i;
        EXPECT_NEAR(pos[i].x, 0.0f, 1e-5f) << i;
        EXPECT_GT(pos[i].y, 1.0f) << i;
        EXPECT_LT(pos[i].y, 1.8f) << i;
    }
}

TEST_F(TreeGaussiansTest, silhouetteMatchesClassical) {
    const std::vector<TreeTypeParams> types = {tree_presets::quakingAspen()};
    const std::vector<TreeInstance> instances = {makeInstance(0, 5, glm::vec3(0.0f), 1.0f / 13.0f)};
    const VkExtent2D extent{384, 384};
    const glm::vec3 eye(0.0f, 0.55f, 1.8f), lookAt(0.0f, 0.55f, 0.0f);

    TreeScene classical(*vc, types, instances, extent, true, true);
    const auto reference = classical.render(eye, lookAt);
    writePpm("compare_classical", reference, extent);

    for (GsplatBackend backend : {GsplatBackend::Raster, GsplatBackend::Compute}) {
        const char* name = backend == GsplatBackend::Raster ? "raster" : "compute";
        SCOPED_TRACE(name);
        const TreeFrame frame = defaultTreeFrame();
        auto generator = std::make_shared<TreeGenerator>(*vc, types, instances,
                                                         TreeCapacity::fromReference(types, instances, frame), frame);
        auto settings = std::make_shared<HostValues<TreeRenderSettings>>(
            *vc, std::vector<TreeRenderSettings>{defaultTreeRenderSettings(extent)});
        auto camera = std::make_shared<CameraUboType>();
        TreeGaussians gaussians(*vc, *generator, camera, settings);
        auto target = std::make_shared<OffscreenTarget>(*vc, extent, 1);
        GsplatConfig config;
        config.shDegree = 0;
        auto splatting = createGaussianSplatting(*vc, backend, target, camera, gaussians.getGaussians(), config);
        ComputeGraph graph(*vc, 1);
        graph.compileFrom(splatting);
        gaussians.zeroHigherBands(1);
        setCamera(camera->ubo, eye, lookAt, 40.0f, 1.0f);
        camera->update(0);
        graph.submitAndWait(vc->getGraphicsQueue(), 0);
        const auto image = readImage(*vc, target->getImage(0), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, extent);
        writePpm(std::string("compare_gaussians_") + name, image, extent);

        // Pixels the splats dim only slightly count as uncovered.
        auto solid = [&](const std::vector<uint8_t>& im, uint32_t x, uint32_t y) {
            const size_t i = (size_t(y) * extent.width + x) * 4;
            return im[i] + im[i + 1] + im[i + 2] > 40;
        };
        size_t both = 0, either = 0;
        for (uint32_t y = 0; y < extent.height; ++y) {
            for (uint32_t x = 0; x < extent.width; ++x) {
                const bool a = solid(reference, x, y), b = solid(image, x, y);
                both += a && b ? 1 : 0;
                either += a || b ? 1 : 0;
            }
        }
        ASSERT_GT(either, 5000u);
        std::cout << name << ": silhouette IoU " << double(both) / double(either) << std::endl;
        // Measured 0.854 (raster) and 0.854 (compute) on Apple M4 / MoltenVK: the splats' soft edges and the
        // classical leaves' level of detail account for the rest.
        EXPECT_GT(double(both) / double(either), 0.8);
    }
}
