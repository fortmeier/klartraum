#include <gtest/gtest.h>
#include <algorithm>
#include <set>

#include "klartraum/headless_frontend.hpp"
#include "klartraum/vulkan_gaussian_splatting.hpp"
#include "klartraum/interface_camera_orbit.hpp"
#include "klartraum/backend_config.hpp"

using namespace klartraum;

// ----------------------------------------------------------------
// Shared test fixture
// ----------------------------------------------------------------
class GaussianSplattingTest : public ::testing::Test {
protected:
    void SetUp() override {
        frontend = std::make_unique<HeadlessFrontend>();
        vulkanContext = &frontend->getKlartraumEngine().getVulkanContext();
    }
    void TearDown() override { frontend.reset(); }

    // Build a camera UBO: camera on the +X axis at distance, looking at origin (Y-up).
    void makeCameraUBO(std::shared_ptr<CameraUboType>& out, float distance = 5.0f) {
        out = std::make_shared<CameraUboType>();
        InterfaceCameraOrbit orbit(InterfaceCameraOrbit::UpDirection::Y);
        orbit.initialize(*vulkanContext);
        orbit.setDistance(distance);
        orbit.update(out->ubo);
    }

    // Build a minimal Gaussian3D (unit scale, identity rotation, given position).
    static Gaussian3D makeGaussian3D(float x, float y, float z, float alpha = 0.5f) {
        Gaussian3D g{};
        g.position = {x, y, z};
        g.scale    = {1.0f, 1.0f, 1.0f};
        g.rotation = {0.0f, 0.0f, 0.0f, 1.0f};
        g.alpha    = alpha;
        return g;
    }

    std::unique_ptr<HeadlessFrontend> frontend;
    VulkanContext* vulkanContext = nullptr;
};

// ----------------------------------------------------------------
// Test 1: projection shader — Gaussian3D → Gaussian2D
//
// Camera sits at (5, 0, 0) looking toward the origin (Y-up).
// In that camera frame:
//   world  Y → camera Y → screen Y (Vulkan-flipped, so Y+1 ends up ABOVE centre)
//   world  Z → camera -X → screen X (negated, so Z+1 ends up LEFT of centre)
//   world  X → camera -Z (depth axis, same screen position as origin)
// ----------------------------------------------------------------
TEST_F(GaussianSplattingTest, projection) {
    const uint32_t N = 3;
    std::vector<Gaussian3D> g3d = {
        makeGaussian3D(0.0f, 0.0f, 0.0f),  // origin → screen centre
        makeGaussian3D(0.0f, 1.0f, 0.0f),  // Y+1    → above centre
        makeGaussian3D(0.0f, 0.0f, 1.0f),  // Z+1    → left of centre
    };

    auto bufIn  = std::make_shared<BufferElementSinglePath<Gaussian3DBuffer>>(*vulkanContext, N);
    auto bufOut = std::make_shared<BufferElement<Gaussian2DBuffer>>(*vulkanContext, N);

    std::shared_ptr<CameraUboType> cameraUBO;
    makeCameraUBO(cameraUBO);

    auto proj = std::make_shared<GaussianProjection>(
        *vulkanContext, "shaders/gsplat/gsplat_projection.comp.spv");
    proj->setInput(bufIn,      0);
    proj->setInput(cameraUBO,  1);
    proj->setInput(bufOut,     2);
    proj->setPushConstants({{N, 4,
        (float)BackendConfig::WIDTH, (float)BackendConfig::HEIGHT}});
    proj->setGroupCountX(N / 128 + 1);

    auto cg = ComputeGraph(*vulkanContext, 1);
    cg.compileFrom(proj);

    bufIn->getBuffer().memcopyFrom(g3d);
    cameraUBO->update(0);

    cg.submitAndWait(vulkanContext->getGraphicsQueue(), 0);

    std::vector<Gaussian2D> out(N);
    bufOut->getBuffer(0).memcopyTo(out);

    const float cx = BackendConfig::WIDTH  * 0.5f;  // 256
    const float cy = BackendConfig::HEIGHT * 0.5f;  // 192

    // Origin projects to the screen centre
    EXPECT_NEAR(out[0].position.x, cx, 3.0f);
    EXPECT_NEAR(out[0].position.y, cy, 3.0f);
    EXPECT_GT(out[0].z, 0.0f);
    EXPECT_LT(out[0].z, 1.0f);

    // Y+1: same screen X, screen Y moves up (smaller Y due to Vulkan flip)
    EXPECT_NEAR(out[1].position.x, cx, 3.0f);
    EXPECT_LT(out[1].position.y, cy);

    // Z+1: screen Y stays centred, screen X moves left (smaller X)
    EXPECT_LT(out[2].position.x, cx);
    EXPECT_NEAR(out[2].position.y, cy, 3.0f);
}

// ----------------------------------------------------------------
// Test 2: binning shader — Gaussian2D → binned Gaussian2D
//
// Four gaussians placed in four distinct bins of a 4×4 grid on a
// 512×384 screen (cell size 128×96). Each gaussian is point-like
// (large covarianceInv → tiny spread) so it falls in exactly one bin.
// ----------------------------------------------------------------
TEST_F(GaussianSplattingTest, binningGaussians) {
    const float W = BackendConfig::WIDTH;   // 512
    const float H = BackendConfig::HEIGHT;  // 384
    const uint32_t gridSize = 4;
    const float cw = W / gridSize;  // 128
    const float ch = H / gridSize;  //  96

    // One gaussian per bin: centres of bins (0,0), (1,0), (0,2), (3,3)
    //   bin index = gridY * gridSize + gridX
    struct BinSpec { float px, py; uint32_t expectedBin; };
    std::vector<BinSpec> specs = {
        { cw * 0.5f, ch * 0.5f,   0 },   // bin  0
        { cw * 1.5f, ch * 0.5f,   1 },   // bin  1
        { cw * 0.5f, ch * 2.5f,   8 },   // bin  8
        { cw * 3.5f, ch * 3.5f,  15 },   // bin 15
    };
    const uint32_t N = (uint32_t)specs.size();

    // Build input: Gaussian2D with z in (0,1) and a point-like covarianceInv
    // (Gaussian2D::covariance is stored as covarianceInv in the shader)
    std::vector<Gaussian2D> input(N);
    for (uint32_t i = 0; i < N; ++i) {
        input[i].position = {specs[i].px, specs[i].py};
        input[i].z        = 0.5f;
        input[i].binMask  = 0;
        input[i].covariance = glm::mat2(1e6f);  // large → point-like spread
        input[i].color    = {1.0f, 1.0f, 1.0f};
        input[i].alpha    = 1.0f;
    }

    // Buffers
    auto bufIn      = std::make_shared<BufferElementSinglePath<Gaussian2DBuffer>>(*vulkanContext, N);
    auto bufOut     = std::make_shared<BufferElement<Gaussian2DBuffer>>(*vulkanContext, N * 2);
    bufOut->setRecordToZero(false);

    auto bufCount   = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(*vulkanContext, 1);
    bufCount->setRecordToZero(true);

    using DispatchBuf = BufferElement<VulkanBuffer<VkDispatchIndirectCommand>>;
    auto bufDispatch = std::make_shared<DispatchBuf>(*vulkanContext, 1);
    bufDispatch->setRecordToZero(true);

    auto bin = std::make_shared<GaussianBinning>(
        *vulkanContext, "shaders/gsplat/gsplat_binning.comp.spv");
    bin->setInput(bufIn,       0);
    bin->setInput(bufOut,      1);
    bin->setInput(bufCount,    2);
    bin->setInput(bufDispatch, 3);
    bin->setPushConstants({{N, gridSize, W, H}});
    bin->setGroupCountX(N / 128 + 1);

    auto cg = ComputeGraph(*vulkanContext, 1);
    cg.compileFrom(bin);

    bufIn->getBuffer().memcopyFrom(input);

    cg.submitAndWait(vulkanContext->getGraphicsQueue(), 0);

    // Read back total count
    std::vector<uint32_t> count(1);
    bufCount->getBuffer(0).memcopyTo(count);
    EXPECT_EQ(count[0], N);

    // Read back binned gaussians and collect binMasks
    std::vector<Gaussian2D> out(N * 2);
    bufOut->getBuffer(0).memcopyTo(out);

    std::set<uint32_t> foundBins;
    for (uint32_t i = 0; i < count[0]; ++i) {
        uint32_t mask = out[i].binMask;
        EXPECT_NE(mask, 0u);
        // Each gaussian should occupy exactly one bin (single bit set)
        EXPECT_EQ(mask & (mask - 1), 0u) << "binMask " << mask << " has more than one bit set";
        foundBins.insert(mask);
    }

    // All four expected bins should be present
    for (const auto& spec : specs) {
        uint32_t expected = 1u << spec.expectedBin;
        EXPECT_TRUE(foundBins.count(expected))
            << "Expected bin " << spec.expectedBin
            << " (mask " << expected << ") not found in output";
    }
}

// ----------------------------------------------------------------
// Test 3: radix sort — ascending integer sort
// ----------------------------------------------------------------
TEST(KlartraumVulkanGaussianSplatting, sort2DGaussians) {
    HeadlessFrontend frontend;

    auto& engine = frontend.getKlartraumEngine();
    auto& vulkanContext = engine.getVulkanContext();

    auto number_streaming_processors = 40;

    const uint32_t number_of_gaussians = 1024 * 1024;

    std::vector<uint32_t> depths;
    std::vector<uint32_t> indexes;
    depths.reserve(number_of_gaussians);
    indexes.reserve(number_of_gaussians);
    for (uint32_t i = 0; i < number_of_gaussians; ++i) {
        depths.push_back(number_of_gaussians - i);
        indexes.push_back(i);
    }

    const uint32_t numBins       = 16;
    const uint32_t threadsPerGroup = 128;
    const uint32_t numWorkGroups = number_streaming_processors * 1024 / threadsPerGroup;

    auto bufValA = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanContext, number_of_gaussians);
    auto bufIdxA = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanContext, number_of_gaussians);
    auto bufValB = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanContext, number_of_gaussians);
    auto bufIdxB = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanContext, number_of_gaussians);

    auto totalCount = std::make_shared<BufferElementSinglePath<VulkanBuffer<uint32_t>>>(vulkanContext, 1);
    totalCount->setName("TotalGaussian2DCounts");
    totalCount->getBuffer(0).memcopyFrom(&number_of_gaussians, 1);

    std::vector<std::string> shaderFiles = {
        "shaders/gsplat/gsplat_radix_sort_histogram.comp.spv",
        "shaders/gsplat/gsplat_radix_sort_hist_prefix_sum.comp.spv",
        "shaders/gsplat/gsplat_radix_sort_hist_scatter.comp.spv"
    };
    auto sortOp = std::make_shared<RadixSort>(vulkanContext, shaderFiles);
    sortOp->setInput(bufValA, 0);
    sortOp->setInput(bufIdxA, 1);
    sortOp->setInput(bufValB, 2);
    sortOp->setInput(bufIdxB, 3);

    auto scratchHistograms = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanContext, numBins * numWorkGroups);
    scratchHistograms->setRecordToZero(true);
    auto scratchCounts  = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanContext, numBins);
    scratchCounts->setRecordToZero(true);
    auto scratchOffsets = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanContext, numWorkGroups + 1);
    scratchOffsets->setRecordToZero(true);

    sortOp->addScratchBufferElement(scratchCounts,     true);
    sortOp->addScratchBufferElement(scratchOffsets,    true);
    sortOp->addScratchBufferElement(totalCount,        false);
    sortOp->addScratchBufferElement(scratchHistograms, true);
    sortOp->setGroupCountX(numWorkGroups);

    const uint32_t passes = 32;
    std::vector<SortPushConstants> pcs;
    for (uint32_t i = 0; i < passes / 4; ++i)
        pcs.push_back({i, number_of_gaussians, numBins});
    sortOp->setPushConstants(pcs);

    auto cg = ComputeGraph(vulkanContext, 1);
    cg.compileFrom(sortOp);

    bufValA->getBuffer(0).memcopyFrom(depths);
    bufIdxA->getBuffer(0).memcopyFrom(indexes);

    cg.submitAndWait(vulkanContext.getGraphicsQueue(), 0);

    std::vector<uint32_t> sortedValues(number_of_gaussians);
    auto outputValues = sortOp->getOutputElement<BufferElement<VulkanBuffer<uint32_t>>>(0);
    outputValues->getBuffer(0).memcopyTo(sortedValues);

    for (uint32_t i = 1; i < (uint32_t)sortedValues.size(); ++i) {
        EXPECT_LE(sortedValues[i-1], sortedValues[i]);
        if (sortedValues[i-1] > sortedValues[i]) break;
    }
}

// ----------------------------------------------------------------
// Test 4: projection → binning chained pipeline
//
// Creates 3D gaussians, projects them, then bins them.
// The projection output feeds directly into the binning shader.
// Verifies that gaussians survive projection (z in [0,1]) and
// end up in the correct screen bins.
// ----------------------------------------------------------------
TEST_F(GaussianSplattingTest, projectionAndBinning) {
    const float W = BackendConfig::WIDTH;
    const float H = BackendConfig::HEIGHT;
    const uint32_t gridSize = 4;

    // Camera at (5, 0, 0) looking at origin.
    // Gaussians placed off-axis so they land in different quadrants.
    // Using large scale (0.01) to make them very small in projected space.
    auto makeSmall = [](float x, float y, float z) {
        Gaussian3D g{};
        g.position = {x, y, z};
        g.scale    = {0.01f, 0.01f, 0.01f};
        g.rotation = {0.0f, 0.0f, 0.0f, 1.0f};
        g.alpha    = 1.0f;
        return g;
    };

    // Place gaussians in known world positions that should project to
    // distinct quadrants of the 512×384 screen (camera on +X axis):
    //   world Y →  screen Y axis (positive world Y = above screen centre)
    //   world Z → -screen X axis (positive world Z = left of screen centre)
    std::vector<Gaussian3D> g3d = {
        makeSmall(0.0f,  0.5f,  0.5f),  // Y+, Z+ → upper-left quadrant
        makeSmall(0.0f,  0.5f, -0.5f),  // Y+, Z- → upper-right quadrant
        makeSmall(0.0f, -0.5f,  0.5f),  // Y-, Z+ → lower-left quadrant
        makeSmall(0.0f, -0.5f, -0.5f),  // Y-, Z- → lower-right quadrant
    };
    const uint32_t N = (uint32_t)g3d.size();
    const uint32_t maxBinned = N * 2;

    auto bufG3D      = std::make_shared<BufferElementSinglePath<Gaussian3DBuffer>>(*vulkanContext, N);
    auto bufG2D      = std::make_shared<BufferElement<Gaussian2DBuffer>>(*vulkanContext, N);
    auto bufBinned   = std::make_shared<BufferElement<Gaussian2DBuffer>>(*vulkanContext, maxBinned);
    auto bufCount    = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(*vulkanContext, 1);
    bufCount->setRecordToZero(true);

    using DispatchBuf = BufferElement<VulkanBuffer<VkDispatchIndirectCommand>>;
    auto bufDispatch = std::make_shared<DispatchBuf>(*vulkanContext, 1);
    bufDispatch->setRecordToZero(true);

    std::shared_ptr<CameraUboType> cameraUBO;
    makeCameraUBO(cameraUBO);

    // Projection stage
    auto projStage = std::make_shared<GaussianProjection>(
        *vulkanContext, "shaders/gsplat/gsplat_projection.comp.spv");
    projStage->setInput(bufG3D,    0);
    projStage->setInput(cameraUBO, 1);
    projStage->setInput(bufG2D,    2);
    projStage->setPushConstants({{N, gridSize, W, H}});
    projStage->setGroupCountX(N / 128 + 1);

    // Binning stage — reads from projection's output buffer (slot 2)
    auto binStage = std::make_shared<GaussianBinning>(
        *vulkanContext, "shaders/gsplat/gsplat_binning.comp.spv");
    binStage->setInput(projStage, 0, 2);  // slot 2 of projStage = bufG2D
    binStage->setInput(bufBinned,    1);
    binStage->setInput(bufCount,     2);
    binStage->setInput(bufDispatch,  3);
    binStage->setPushConstants({{N, gridSize, W, H}});
    binStage->setGroupCountX(N / 128 + 1);

    auto cg = ComputeGraph(*vulkanContext, 1);
    cg.compileFrom(binStage);

    bufG3D->getBuffer().memcopyFrom(g3d);
    cameraUBO->update(0);

    cg.submitAndWait(vulkanContext->getGraphicsQueue(), 0);

    // All 4 gaussians should survive projection (z in [0,1]) and be binned
    std::vector<uint32_t> count(1);
    bufCount->getBuffer(0).memcopyTo(count);
    EXPECT_EQ(count[0], N) << "Expected all " << N << " gaussians to be binned";

    // Each output gaussian must have exactly one bin bit set
    std::vector<Gaussian2D> binned(maxBinned);
    bufBinned->getBuffer(0).memcopyTo(binned);
    std::set<uint32_t> foundBins;
    for (uint32_t i = 0; i < count[0]; ++i) {
        uint32_t mask = binned[i].binMask;
        EXPECT_NE(mask, 0u);
        EXPECT_EQ(mask & (mask - 1), 0u) << "binMask has multiple bits set";
        foundBins.insert(mask);
    }
    // All 4 gaussians in different bins
    EXPECT_EQ(foundBins.size(), N);
}
