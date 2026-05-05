#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <set>

#include "klartraum/headless_frontend.hpp"
#include "klartraum/vulkan_gaussian_splatting.hpp"
#include "klartraum/interface_camera_orbit.hpp"
#include "klartraum/backend_config.hpp"
#include "klartraum/computegraph/imageviewsrc.hpp"

#include "load-spz.h"

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

// ----------------------------------------------------------------
// Shared pipeline helper for the single-gaussian and SPZ-file tests.
// Runs: extract sort keys → radix sort → gather sorted →
//       compute bounds → splatting.
// Inputs bufBinned/bufCount/imageViewSrc must already be populated.
// ----------------------------------------------------------------
struct Phase2Params {
    uint32_t totalCount;
    uint32_t numSortWorkGroups;
    uint32_t gridSize;
    float    W, H;
};

static void runSortBoundsSplat(
    VulkanContext&              vc,
    Phase2Params                p,
    std::shared_ptr<BufferElement<Gaussian2DBuffer>>         bufBinned,
    std::shared_ptr<BufferElement<VulkanBuffer<uint32_t>>>   bufCount,
    std::shared_ptr<ImageViewSrc>                            imageViewSrc)
{
    const uint32_t numBins     = p.gridSize * p.gridSize;
    const uint32_t maxBinned   = bufBinned->getBuffer(0).getSize();
    const uint32_t tpg         = 128;

    auto bufValA = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vc, maxBinned);
    auto bufIdxA = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vc, maxBinned);
    auto bufValB = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vc, maxBinned);
    auto bufIdxB = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vc, maxBinned);

    auto scratchHist    = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vc, numBins * p.numSortWorkGroups);
    scratchHist->setRecordToZero(true);
    auto scratchCounts  = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vc, numBins);
    scratchCounts->setRecordToZero(true);
    auto scratchOffsets = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vc, p.numSortWorkGroups + 1);
    scratchOffsets->setRecordToZero(true);

    auto bufSorted = std::make_shared<BufferElement<Gaussian2DBuffer>>(vc, maxBinned);
    auto bufBounds = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vc, numBins * 2);
    bufBounds->setRecordToZero(true);

    auto extractStage = std::make_shared<GeneralComputation<>>(
        vc, "shaders/gsplat/gsplat_extract_sort_keys.comp.spv");
    extractStage->setInput(bufBinned, 0);
    extractStage->setInput(bufCount,  1);
    extractStage->setInput(bufValA,   2);
    extractStage->setInput(bufIdxA,   3);
    extractStage->setGroupCountX(p.totalCount / tpg + 1);

    std::vector<std::string> sortShaders = {
        "shaders/gsplat/gsplat_radix_sort_histogram.comp.spv",
        "shaders/gsplat/gsplat_radix_sort_hist_prefix_sum.comp.spv",
        "shaders/gsplat/gsplat_radix_sort_hist_scatter.comp.spv"
    };
    auto sortOp = std::make_shared<RadixSort>(vc, sortShaders);
    sortOp->setInput(bufValA, 0); sortOp->setInput(bufIdxA, 1);
    sortOp->setInput(bufValB, 2); sortOp->setInput(bufIdxB, 3);
    sortOp->addScratchBufferElement(scratchCounts,  true);
    sortOp->addScratchBufferElement(scratchOffsets, true);
    sortOp->addScratchBufferElement(bufCount,       false);
    sortOp->addScratchBufferElement(scratchHist,    true);
    sortOp->setGroupCountX(p.numSortWorkGroups);
    {
        std::vector<SortPushConstants> pcs;
        for (uint32_t i = 0; i < 32 / 4; ++i)
            pcs.push_back({i, p.totalCount, numBins});
        sortOp->setPushConstants(pcs);
    }

    // After 8 passes (0-7), last pass=7 (odd) → sorted output in A buffers
    auto gatherStage = std::make_shared<GeneralComputation<>>(
        vc, "shaders/gsplat/gsplat_gather_sorted.comp.spv");
    gatherStage->setInput(bufBinned, 0);
    gatherStage->setInput(bufIdxA,   1);
    gatherStage->setInput(bufCount,  2);
    gatherStage->setInput(bufSorted, 3);
    gatherStage->setGroupCountX(p.totalCount / tpg + 1);

    auto boundsStage = std::make_shared<GaussianComputeBounds>(
        vc, "shaders/gsplat/gsplat_bin_bounds.comp.spv");
    boundsStage->setInput(bufSorted, 0);
    boundsStage->setInput(bufCount,  1);
    boundsStage->setInput(bufBounds, 2);
    boundsStage->setPushConstants({{p.totalCount, p.gridSize, p.W, p.H}});
    boundsStage->setGroupCountX(p.totalCount / 256 + 1);

    const uint32_t tbX = 8, tbY = 8;
    const uint32_t gpbX = uint32_t((p.W / tbX) / p.gridSize);
    const uint32_t gpbY = uint32_t((p.H / tbY) / p.gridSize);

    auto splatStage = std::make_shared<GaussianSplatting>(
        vc, "shaders/gsplat/gsplat_binned_splatting.comp.spv");
    splatStage->setInput(bufSorted,    0);
    splatStage->setInput(bufCount,     1);
    splatStage->setInput(bufBounds,    2);
    splatStage->setInput(imageViewSrc, 3);
    {
        std::vector<SplatPushConstants> pcs;
        for (uint32_t y = 0; y < p.gridSize; ++y)
            for (uint32_t x = 0; x < p.gridSize; ++x)
                pcs.push_back({p.totalCount, p.gridSize, x, y, p.W, p.H});
        splatStage->setPushConstants(pcs);
    }
    splatStage->setGroupCountX(gpbX);
    splatStage->setGroupCountY(gpbY);
    splatStage->setGroupCountZ(1);

    // Each stage is submitted in its own compute graph so that stages that
    // share buffer objects (e.g. bufValA written by extract, read by sort)
    // actually execute in order.  A single compileFrom(splatStage) would
    // only include stages that are connected via setInput chains; since
    // these stages communicate through shared buffer pointers rather than
    // graph edges, they must be driven separately.
    //
    // IMPORTANT: buffers with setRecordToZero(true) are zeroed at the start
    // of *every* CG they participate in.  bufCount must not be zeroed after
    // phase 1 fills it, and bufBounds must not be zeroed after boundsStage
    // fills it.  Disable auto-zero before the stages that read these values.
    bufCount->setRecordToZero(false);  // phase 1 already zeroed + filled it

    auto submit = [&](auto stage) {
        auto cg = ComputeGraph(vc, 1);
        cg.compileFrom(stage);
        cg.submitAndWait(vc.getGraphicsQueue(), 0);
    };
    submit(extractStage);
    submit(sortOp);
    submit(gatherStage);
    // boundsStage needs bufBounds zeroed before it runs (setRecordToZero(true)
    // handles that); after it fills the bounds, disable auto-zero so splatStage
    // does not clear the result.
    submit(boundsStage);
    bufBounds->setRecordToZero(false);
    submit(splatStage);
}

// ----------------------------------------------------------------
// Helper: write a BGRA image (4 bytes/pixel, row-major) to a binary
// PPM (P6) file. PPM requires RGB so B and R channels are swapped.
// ----------------------------------------------------------------
static void writePPM(const std::string& path,
                     const uint8_t* bgra,
                     uint32_t width, uint32_t height)
{
    std::ofstream f(path, std::ios::binary);
    f << "P6\n" << width << " " << height << "\n255\n";
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            const uint8_t* p = bgra + (y * width + x) * 4;
            uint8_t rgb[3] = { p[2], p[1], p[0] };  // BGRA → RGB
            f.write(reinterpret_cast<const char*>(rgb), 3);
        }
    }
}

// ----------------------------------------------------------------
// Helper: load an SPZ file and convert to Gaussian3D host data.
// Mirrors the logic in VulkanGaussianSplatting::loadSPZModel.
// ----------------------------------------------------------------
static std::vector<Gaussian3D> loadSpz(const std::string& path) {
    spz::PackedGaussians packed = spz::loadSpzPacked(path);
    spz::CoordinateConverter conv;
    std::vector<Gaussian3D> out;
    out.reserve(packed.numPoints);
    for (int i = 0; i < packed.numPoints; ++i) {
        spz::UnpackedGaussian g = packed.unpack(i, conv);
        Gaussian3D g3d;
        std::memcpy(&g3d, &g, sizeof(spz::UnpackedGaussian));
        g3d.alpha    = 1.0f / (1.0f + std::exp(-g.alpha));
        g3d.scale[0] = std::exp(g.scale[0]);
        g3d.scale[1] = std::exp(g.scale[1]);
        g3d.scale[2] = std::exp(g.scale[2]);
        out.push_back(g3d);
    }
    return out;
}

// ----------------------------------------------------------------
// Helper: copy a VkImage (BGRA GENERAL layout) to a host buffer
// and write it as a binary PPM (P6) file. B and R are swapped.
// ----------------------------------------------------------------
static void saveImageAsPPM(VulkanContext& vc,
                           VkImage image,
                           uint32_t W, uint32_t H,
                           const std::string& path)
{
    const VkDeviceSize bytes = W * H * 4;
    VkBuffer       buf; VkDeviceMemory mem;
    vc.createBuffer(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    buf, mem);

    VkCommandBufferAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool        = vc.getCommandPool();
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
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
    region.imageExtent = {W, H, 1};
    vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_GENERAL, buf, 1, &region);
    vkEndCommandBuffer(cmd);

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    vkQueueSubmit(vc.getGraphicsQueue(), 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(vc.getGraphicsQueue());
    vkFreeCommandBuffers(vc.getDevice(), vc.getCommandPool(), 1, &cmd);

    void* data;
    vkMapMemory(vc.getDevice(), mem, 0, bytes, 0, &data);
    writePPM(path, static_cast<const uint8_t*>(data), W, H);
    vkUnmapMemory(vc.getDevice(), mem);
    vkFreeMemory(vc.getDevice(), mem, nullptr);
    vkDestroyBuffer(vc.getDevice(), buf, nullptr);
}

// ----------------------------------------------------------------
// Test 5: full gaussian splatting pipeline (headless offscreen)
//
// Pipeline (using the same shader setup as the individual unit tests):
//   projection → binning → extract sort keys → radix sort →
//   gather sorted → compute bin bounds → splatting
//
// Phase 1: projection + binning.  Wait and read totalCount from GPU.
// Phase 2: extract → sort → gather → bounds → splat.  Wait.
// Then copy the rendered offscreen image to host and write a PPM file.
// ----------------------------------------------------------------
TEST_F(GaussianSplattingTest, renderWithSpzFile) {
    const std::string spzPath = "3rdparty/spz/samples/racoonfamily.spz";
    if (!std::filesystem::exists(spzPath)) {
        GTEST_SKIP() << "SPZ sample not found: " << spzPath;
    }

    // ---- Load SPZ ----
    auto g3dHost = loadSpz(spzPath);
    const uint32_t N = (uint32_t)g3dHost.size();
    const uint32_t maxBinned = N * 2;
    std::cout << "Loaded " << N << " gaussians" << std::endl;

    const float W = BackendConfig::WIDTH;
    const float H = BackendConfig::HEIGHT;
    const uint32_t gridSize = 4;
    const uint32_t numBins  = gridSize * gridSize;

    // Sort constants (same as sort2DGaussians unit test)
    const uint32_t threadsPerGroup   = 128;
    const uint32_t streamingProcs    = 40;
    const uint32_t numSortWorkGroups = streamingProcs * 1024 / threadsPerGroup; // 320

    // ---- Shared buffers ----
    auto bufG3D    = std::make_shared<BufferElementSinglePath<Gaussian3DBuffer>>(*vulkanContext, N);
    auto bufG2D    = std::make_shared<BufferElement<Gaussian2DBuffer>>(*vulkanContext, N);
    auto bufBinned = std::make_shared<BufferElement<Gaussian2DBuffer>>(*vulkanContext, maxBinned);
    bufBinned->setRecordToZero(false);

    auto bufCount  = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(*vulkanContext, 1);
    bufCount->setRecordToZero(true);

    using DispatchBuf = BufferElement<VulkanBuffer<VkDispatchIndirectCommand>>;
    auto bufDispatch = std::make_shared<DispatchBuf>(*vulkanContext, 1);
    bufDispatch->setRecordToZero(true);

    std::shared_ptr<CameraUboType> cameraUBO;
    makeCameraUBO(cameraUBO, 2.0f);

    // ---- Phase 1: projection + binning ----
    auto projStage = std::make_shared<GaussianProjection>(
        *vulkanContext, "shaders/gsplat/gsplat_projection.comp.spv");
    projStage->setInput(bufG3D,    0);
    projStage->setInput(cameraUBO, 1);
    projStage->setInput(bufG2D,    2);
    projStage->setPushConstants({{N, gridSize, W, H}});
    projStage->setGroupCountX(N / threadsPerGroup + 1);

    auto binStage = std::make_shared<GaussianBinning>(
        *vulkanContext, "shaders/gsplat/gsplat_binning.comp.spv");
    binStage->setInput(projStage,   0, 2);  // slot 2 of proj = bufG2D
    binStage->setInput(bufBinned,   1);
    binStage->setInput(bufCount,    2);
    binStage->setInput(bufDispatch, 3);
    binStage->setPushConstants({{N, gridSize, W, H}});
    binStage->setGroupCountX(N / threadsPerGroup + 1);

    {
        auto cg1 = ComputeGraph(*vulkanContext, 1);
        cg1.compileFrom(binStage);
        bufG3D->getBuffer().memcopyFrom(g3dHost);
        cameraUBO->update(0);
        cg1.submitAndWait(vulkanContext->getGraphicsQueue(), 0);
    }

    // Read totalCount from GPU
    uint32_t totalCount = 0;
    {
        std::vector<uint32_t> tmp(1);
        bufCount->getBuffer(0).memcopyTo(tmp);
        totalCount = tmp[0];
    }
    ASSERT_GT(totalCount, 0u) << "Binning produced zero gaussians";
    std::cout << "Binned " << totalCount << " gaussians" << std::endl;

    // ---- Phase 2: extract → sort → gather → bounds → splat ----

    // Sort buffers (uint32_t: sort values and indices, ping-pong A/B)
    auto bufValA = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(*vulkanContext, maxBinned);
    auto bufIdxA = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(*vulkanContext, maxBinned);
    auto bufValB = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(*vulkanContext, maxBinned);
    auto bufIdxB = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(*vulkanContext, maxBinned);

    // Scratch buffers for radix sort
    auto scratchHist    = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(*vulkanContext, numBins * numSortWorkGroups);
    scratchHist->setRecordToZero(true);
    auto scratchCounts  = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(*vulkanContext, numBins);
    scratchCounts->setRecordToZero(true);
    auto scratchOffsets = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(*vulkanContext, numSortWorkGroups + 1);
    scratchOffsets->setRecordToZero(true);

    // Sorted Gaussian2D output (gather result) and bounds
    auto bufSorted = std::make_shared<BufferElement<Gaussian2DBuffer>>(*vulkanContext, maxBinned);
    auto bufBounds = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(*vulkanContext, numBins * 2);
    bufBounds->setRecordToZero(true);

    // Stage: extract binMask → radixValuesA, indexes 0..N-1 → indexesA
    auto extractStage = std::make_shared<GeneralComputation<>>(
        *vulkanContext, "shaders/gsplat/gsplat_extract_sort_keys.comp.spv");
    extractStage->setInput(bufBinned, 0);
    extractStage->setInput(bufCount,  1);
    extractStage->setInput(bufValA,   2);
    extractStage->setInput(bufIdxA,   3);
    extractStage->setGroupCountX(totalCount / threadsPerGroup + 1);

    // Stage: radix sort (same setup as sort2DGaussians unit test)
    // 8 passes over 32 bits; last pass=7 (odd) → output in A buffers
    std::vector<std::string> sortShaders = {
        "shaders/gsplat/gsplat_radix_sort_histogram.comp.spv",
        "shaders/gsplat/gsplat_radix_sort_hist_prefix_sum.comp.spv",
        "shaders/gsplat/gsplat_radix_sort_hist_scatter.comp.spv"
    };
    auto sortOp = std::make_shared<RadixSort>(*vulkanContext, sortShaders);
    sortOp->setInput(bufValA, 0);
    sortOp->setInput(bufIdxA, 1);
    sortOp->setInput(bufValB, 2);
    sortOp->setInput(bufIdxB, 3);
    sortOp->addScratchBufferElement(scratchCounts,  true);
    sortOp->addScratchBufferElement(scratchOffsets, true);
    sortOp->addScratchBufferElement(bufCount,       false);
    sortOp->addScratchBufferElement(scratchHist,    true);
    sortOp->setGroupCountX(numSortWorkGroups);
    {
        std::vector<SortPushConstants> pcs;
        for (uint32_t i = 0; i < 32 / 4; ++i)
            pcs.push_back({i, totalCount, numBins});
        sortOp->setPushConstants(pcs);
    }

    // Stage: gather sorted Gaussian2D using sorted indexesA
    auto gatherStage = std::make_shared<GeneralComputation<>>(
        *vulkanContext, "shaders/gsplat/gsplat_gather_sorted.comp.spv");
    gatherStage->setInput(bufBinned, 0);  // unsorted source
    gatherStage->setInput(bufIdxA,   1);  // sorted indices (output of sort is in A after 8 passes)
    gatherStage->setInput(bufCount,  2);
    gatherStage->setInput(bufSorted, 3);
    gatherStage->setGroupCountX(totalCount / threadsPerGroup + 1);

    // Stage: compute bin start/end bounds
    auto boundsStage = std::make_shared<GaussianComputeBounds>(
        *vulkanContext, "shaders/gsplat/gsplat_bin_bounds.comp.spv");
    boundsStage->setInput(bufSorted, 0);
    boundsStage->setInput(bufCount,  1);
    boundsStage->setInput(bufBounds, 2);
    boundsStage->setPushConstants({{totalCount, gridSize, W, H}});
    boundsStage->setGroupCountX(totalCount / 256 + 1);

    // Stage: splatting — one dispatch per bin (16 bins for 4×4 grid)
    // Each dispatch covers groupsPerBinX×groupsPerBinY workgroups of 8×8 threads.
    const uint32_t tbX = 8, tbY = 8;
    const uint32_t gpbX = uint32_t((W / tbX) / gridSize);  // 16
    const uint32_t gpbY = uint32_t((H / tbY) / gridSize);  // 12

    // Build ImageViewSrc with the headless offscreen image
    uint32_t numImages = vulkanContext->getNumberOfSwapChainImages();
    VkExtent2D extent  = vulkanContext->getSwapChainExtent();
    std::vector<VkImageView> ivs(numImages);
    std::vector<VkImage>     imgs(numImages);
    std::vector<VkExtent2D>  exts(numImages, extent);
    for (uint32_t i = 0; i < numImages; ++i) {
        ivs[i]  = vulkanContext->getImageView(i);
        imgs[i] = vulkanContext->getSwapChainImage(i);
    }
    auto imageViewSrc = std::make_shared<ImageViewSrc>(ivs, imgs, exts);

    auto splatStage = std::make_shared<GaussianSplatting>(
        *vulkanContext, "shaders/gsplat/gsplat_binned_splatting.comp.spv");
    splatStage->setInput(bufSorted,    0);
    splatStage->setInput(bufCount,     1);
    splatStage->setInput(bufBounds,    2);
    splatStage->setInput(imageViewSrc, 3);
    {
        std::vector<SplatPushConstants> pcs;
        for (uint32_t y = 0; y < gridSize; ++y)
            for (uint32_t x = 0; x < gridSize; ++x)
                pcs.push_back({totalCount, gridSize, x, y, W, H});
        splatStage->setPushConstants(pcs);
    }
    splatStage->setGroupCountX(gpbX);
    splatStage->setGroupCountY(gpbY);
    splatStage->setGroupCountZ(1);

    // Submit each stage separately; disable auto-zero on shared buffers
    // after the stage that writes them, so subsequent stages see the data.
    bufCount->setRecordToZero(false);  // phase 1 already filled it
    auto submit2 = [&](auto stage) {
        auto cg = ComputeGraph(*vulkanContext, 1);
        cg.compileFrom(stage);
        cg.submitAndWait(vulkanContext->getGraphicsQueue(), 0);
    };
    submit2(extractStage);
    submit2(sortOp);
    submit2(gatherStage);
    submit2(boundsStage);
    bufBounds->setRecordToZero(false);
    submit2(splatStage);

    // ---- Copy rendered image to host and write PPM ----
    saveImageAsPPM(*vulkanContext, imgs[0], (uint32_t)W, (uint32_t)H,
                   "test_gaussian_splatting_render.ppm");

    SUCCEED() << "Rendered " << (uint32_t)W << "x" << (uint32_t)H
              << " image written to test_gaussian_splatting_render.ppm";
}

// ----------------------------------------------------------------
// Test 6: single red gaussian — minimal pipeline smoke test
//
// Creates one Gaussian3D at the world origin with a color that
// produces red in the output PPM (accounting for the BGRA↔RGBA
// swap between the rgba8 storage image and the host readback).
// Camera sits at (5, 0, 0) looking at the origin, so the gaussian
// projects to the screen centre ~(256, 192).
//
// After rendering, the test verifies:
//   • At least one pixel in a window around the screen centre has a
//     channel value > 200 (the gaussian was drawn and is bright).
//   • That same region's maximum channel is in the blue byte position
//     (byte 0 in BGRA host memory), which the PPM writer converts to
//     red — so the file looks red.
// ----------------------------------------------------------------
TEST_F(GaussianSplattingTest, singleRedGaussian) {
    const float    W        = BackendConfig::WIDTH;   // 512
    const float    H        = BackendConfig::HEIGHT;  // 384
    const uint32_t gridSize = 4;
    const uint32_t numBins  = gridSize * gridSize;    // 16

    // ---- Place the gaussian at the CENTRE of bin 10 (gridX=2, gridY=2) ----
    // Bin 10 covers x=[256,384), y=[192,288) → centre (320,240) in screen space.
    //
    // Camera: lookAt(eye=(5,0,0), centre=(0,0,0), Y-up).
    // View matrix maps world(x,y,z) → camera(cx=-z, cy=y, cz=x-5).
    // NDC formula: ndcX = focalX*cx/(-cz), ndcY = focalY*cy/(-cz)
    //   (focalX≈2.414, focalY≈2.414/aspect≈1.811)
    // Screen: sx=(ndcX+1)*W/2, sy=(ndcY+1)*H/2.
    // For (sx,sy)=(320,240): ndcX=0.25, ndcY=0.25.
    // Solve: cx=0.25*5/2.414≈0.518, cy=0.25*(-5)/1.811≈-0.690, cz=-5
    //   → world.x=0, world.y=-0.690, world.z=-0.518
    //
    // scale=0.05 → projected σ≈12px, spread≈30px.  Bin cells are 128×96px,
    // so the gaussian is ≥34px from every edge of bin 10 → overlaps only bin 10.
    // totalCount = 1, known at compile time.
    //
    // SH colour: 0.5 + base * 0.282095 = target  →  base=(target−0.5)/0.282095
    //   r_shader=1.0  →  base_R = 1.772   (shader writes r=1 → byte2=255 → PPM red)
    //   g_shader=0.0  →  base_G = -1.772
    //   b_shader=0.0  →  base_B = -1.772
    const float SH_ONE = 1.772f, SH_ZERO = -1.772f;
    Gaussian3D g{};
    g.position = {0.0f, -0.690f, -0.518f};  // projects to screen (320,240) = bin 10 centre
    g.scale    = {0.05f, 0.05f, 0.05f};     // small → only overlaps bin 10
    g.rotation = {0.0f, 0.0f, 0.0f, 1.0f};
    g.color    = {SH_ONE, SH_ZERO, SH_ZERO};
    g.alpha    = 1.0f;

    // ---- Shared constants ----
    const uint32_t N           = 1;                    // exactly 1 gaussian
    const uint32_t maxBinned   = 4;                    // at most 4 bin-copies (N*2 rule with N=2)
    const uint32_t nsg         = 1;                    // 1 sort workgroup (numElements=1)
    const uint32_t tpg         = 128;

    // ---- Shared buffers ----
    auto bufG3D    = std::make_shared<BufferElementSinglePath<Gaussian3DBuffer>>(*vulkanContext, N);
    auto bufG2D    = std::make_shared<BufferElement<Gaussian2DBuffer>>(*vulkanContext, N);
    auto bufBinned = std::make_shared<BufferElement<Gaussian2DBuffer>>(*vulkanContext, maxBinned);
    bufBinned->setRecordToZero(false);
    auto bufCount  = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(*vulkanContext, 1);
    bufCount->setRecordToZero(true);
    using DispBuf  = BufferElement<VulkanBuffer<VkDispatchIndirectCommand>>;
    auto bufDisp   = std::make_shared<DispBuf>(*vulkanContext, 1);
    bufDisp->setRecordToZero(true);

    // Sort ping-pong + scratch
    auto bufValA = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(*vulkanContext, maxBinned);
    auto bufIdxA = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(*vulkanContext, maxBinned);
    auto bufValB = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(*vulkanContext, maxBinned);
    auto bufIdxB = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(*vulkanContext, maxBinned);
    auto scratchHist    = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(*vulkanContext, numBins * nsg);
    scratchHist->setRecordToZero(true);
    auto scratchCounts  = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(*vulkanContext, numBins);
    scratchCounts->setRecordToZero(true);
    auto scratchOffsets = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(*vulkanContext, nsg + 1);
    scratchOffsets->setRecordToZero(true);

    auto bufSorted = std::make_shared<BufferElement<Gaussian2DBuffer>>(*vulkanContext, maxBinned);
    auto bufBounds = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(*vulkanContext, numBins * 2);
    bufBounds->setRecordToZero(true);

    // ---- ImageViewSrc ----
    uint32_t numImages = vulkanContext->getNumberOfSwapChainImages();
    VkExtent2D ext = vulkanContext->getSwapChainExtent();
    std::vector<VkImageView> ivs(numImages); std::vector<VkImage> imgs(numImages);
    std::vector<VkExtent2D> exts(numImages, ext);
    for (uint32_t i = 0; i < numImages; ++i) {
        ivs[i] = vulkanContext->getImageView(i); imgs[i] = vulkanContext->getSwapChainImage(i);
    }
    auto imageViewSrc = std::make_shared<ImageViewSrc>(ivs, imgs, exts);

    std::shared_ptr<CameraUboType> cameraUBO;
    makeCameraUBO(cameraUBO);

    // ---- Stage setup ----
    // projStage: Gaussian3D → Gaussian2D
    auto projStage = std::make_shared<GaussianProjection>(
        *vulkanContext, "shaders/gsplat/gsplat_projection.comp.spv");
    projStage->setInput(bufG3D, 0); projStage->setInput(cameraUBO, 1); projStage->setInput(bufG2D, 2);
    projStage->setPushConstants({{N, gridSize, W, H}});
    projStage->setGroupCountX(1);

    // binStage: Gaussian2D → binned Gaussian2D + count
    // numElements=N+1 so output limit = (N+1)*2 = 4 = maxBinned, covering all bin-copies.
    auto binStage = std::make_shared<GaussianBinning>(
        *vulkanContext, "shaders/gsplat/gsplat_binning.comp.spv");
    binStage->setInput(projStage, 0, 2);   // input 0 = bufG2D (slot 2 of projStage)
    binStage->setInput(bufBinned, 1);
    binStage->setInput(bufCount,  2);
    binStage->setInput(bufDisp,   3);
    binStage->setPushConstants({{N, gridSize, W, H}});  // only process index 0; limit = N*2 = 2
    binStage->setGroupCountX(1);

    // extractStage: binnedGaussians → sort keys (binMask) + indexes 0..N
    auto extractStage = std::make_shared<GeneralComputation<>>(
        *vulkanContext, "shaders/gsplat/gsplat_extract_sort_keys.comp.spv");
    extractStage->setInput(binStage, 0, 1);  // input 0 = bufBinned (slot 1 of binStage)
    extractStage->setInput(binStage, 1, 2);  // input 1 = bufCount  (slot 2 of binStage)
    extractStage->setInput(bufValA, 2);
    extractStage->setInput(bufIdxA, 3);
    extractStage->setGroupCountX(1);

    // sortOp: radix sort (8 passes, 4 bits each)
    std::vector<std::string> sortShaders = {
        "shaders/gsplat/gsplat_radix_sort_histogram.comp.spv",
        "shaders/gsplat/gsplat_radix_sort_hist_prefix_sum.comp.spv",
        "shaders/gsplat/gsplat_radix_sort_hist_scatter.comp.spv"
    };
    auto sortOp = std::make_shared<RadixSort>(*vulkanContext, sortShaders);
    sortOp->setInput(extractStage, 0, 2);  // bufValA (slot 2)
    sortOp->setInput(extractStage, 1, 3);  // bufIdxA (slot 3)
    sortOp->setInput(bufValB, 2); sortOp->setInput(bufIdxB, 3);
    sortOp->addScratchBufferElement(scratchCounts,  true);
    sortOp->addScratchBufferElement(scratchOffsets, true);
    sortOp->addScratchBufferElement(bufCount,       false);  // passes numElements to sort
    sortOp->addScratchBufferElement(scratchHist,    true);
    sortOp->setGroupCountX(nsg);
    {
        std::vector<SortPushConstants> pcs;
        for (uint32_t i = 0; i < 32 / 4; ++i)
            pcs.push_back({i, N, numBins});  // numElements=1
        sortOp->setPushConstants(pcs);
    }

    // gatherStage: reorder binnedGaussians by sorted indices
    // After 8 passes (last=7, odd), sorted output is in A buffers.
    auto gatherStage = std::make_shared<GeneralComputation<>>(
        *vulkanContext, "shaders/gsplat/gsplat_gather_sorted.comp.spv");
    gatherStage->setInput(binStage, 0, 1);  // bufBinned
    gatherStage->setInput(sortOp, 1, 1);    // bufIdxA (sorted, via extractStage slot 3)
    gatherStage->setInput(binStage, 2, 2);  // bufCount
    gatherStage->setInput(bufSorted, 3);
    gatherStage->setGroupCountX(1);

    // boundsStage: compute bin start/end ranges
    auto boundsStage = std::make_shared<GaussianComputeBounds>(
        *vulkanContext, "shaders/gsplat/gsplat_bin_bounds.comp.spv");
    boundsStage->setInput(gatherStage, 0, 3);  // bufSorted
    boundsStage->setInput(binStage, 1, 2);     // bufCount
    boundsStage->setInput(bufBounds, 2);
    boundsStage->setPushConstants({{N, gridSize, W, H}});
    boundsStage->setGroupCountX(1);

    // splatStage: render gaussians to image
    const uint32_t tbX = 8, tbY = 8;
    const uint32_t gpbX = (uint32_t)(W / tbX) / gridSize;
    const uint32_t gpbY = (uint32_t)(H / tbY) / gridSize;
    auto splatStage = std::make_shared<GaussianSplatting>(
        *vulkanContext, "shaders/gsplat/gsplat_binned_splatting.comp.spv");
    splatStage->setInput(gatherStage, 0, 3);  // bufSorted
    splatStage->setInput(binStage, 1, 2);     // bufCount
    splatStage->setInput(boundsStage, 2, 2);  // bufBounds
    splatStage->setInput(imageViewSrc, 3);
    {
        std::vector<SplatPushConstants> pcs;
        for (uint32_t y = 0; y < gridSize; ++y)
            for (uint32_t x = 0; x < gridSize; ++x)
                pcs.push_back({N, gridSize, x, y, W, H});
        splatStage->setPushConstants(pcs);
    }
    splatStage->setGroupCountX(gpbX);
    splatStage->setGroupCountY(gpbY);
    splatStage->setGroupCountZ(1);

    // ---- Single compute graph: compileFrom traverses the full chain ----
    auto cg = ComputeGraph(*vulkanContext, 1);
    cg.compileFrom(splatStage);

    bufG3D->getBuffer().memcopyFrom({g});
    cameraUBO->update(0);

    cg.submitAndWait(vulkanContext->getGraphicsQueue(), 0);

    // ---- Read back and verify ----
    saveImageAsPPM(*vulkanContext, imgs[0], (uint32_t)W, (uint32_t)H,
                   "test_single_red_gaussian.ppm");

    const VkDeviceSize bytes = (uint32_t)W * (uint32_t)H * 4;
    VkBuffer stBuf; VkDeviceMemory stMem;
    vulkanContext->createBuffer(bytes,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        stBuf, stMem);
    {
        VkCommandBufferAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool = vulkanContext->getCommandPool();
        ai.level       = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VkCommandBuffer cmd;
        vkAllocateCommandBuffers(vulkanContext->getDevice(), &ai, &cmd);
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &bi);
        VkBufferImageCopy region{};
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = {(uint32_t)W, (uint32_t)H, 1};
        vkCmdCopyImageToBuffer(cmd, imgs[0], VK_IMAGE_LAYOUT_GENERAL, stBuf, 1, &region);
        vkEndCommandBuffer(cmd);
        VkSubmitInfo si{}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
        vkQueueSubmit(vulkanContext->getGraphicsQueue(), 1, &si, VK_NULL_HANDLE);
        vkQueueWaitIdle(vulkanContext->getGraphicsQueue());
        vkFreeCommandBuffers(vulkanContext->getDevice(),
                             vulkanContext->getCommandPool(), 1, &cmd);
    }
    void* data;
    vkMapMemory(vulkanContext->getDevice(), stMem, 0, bytes, 0, &data);
    const uint8_t* px = static_cast<const uint8_t*>(data);

    // Gaussian projected to (320,240) = centre of bin 10.
    // byte2 = r_shader → PPM red (bright), byte0 = b_shader → PPM blue (low).
    const uint32_t gcx = 320, gcy = 240, radius = 20;
    uint8_t maxByte2 = 0, maxByte0 = 0;
    for (uint32_t y = gcy - radius; y <= gcy + radius; ++y)
        for (uint32_t x = gcx - radius; x <= gcx + radius; ++x) {
            const uint8_t* p = px + (y * (uint32_t)W + x) * 4;
            if (p[2] > maxByte2) maxByte2 = p[2];
            if (p[0] > maxByte0) maxByte0 = p[0];
        }

    vkUnmapMemory(vulkanContext->getDevice(), stMem);
    vkFreeMemory(vulkanContext->getDevice(), stMem, nullptr);
    vkDestroyBuffer(vulkanContext->getDevice(), stBuf, nullptr);

    EXPECT_GT(maxByte2, uint8_t(200))
        << "byte2 (r_shader → PPM red) should be bright near gaussian centre (320,240)";
    EXPECT_LT(maxByte0, uint8_t(50))
        << "byte0 (b_shader → PPM blue) should be low";

    SUCCEED() << "Rendered single red gaussian to test_single_red_gaussian.ppm";
}

// ----------------------------------------------------------------
// Test 7: splatting shader in isolation with pre-computed inputs
//
// Bypasses projection / binning / sort / bounds entirely.
// Directly provides:
//   • One Gaussian2D at screen centre, covarianceInv gives ~20 px σ,
//     color=(0,0,1) in shader RGBA → stored as BGRA byte2=255 → PPM R=255
//   • totalCount = 1
//   • StartAndEnd table: only bin 10 (gridX=2, gridY=2) has [0,1)
//
// Then runs only the splatting shader, reads back the image, and checks:
//   • A pixel near screen centre has R channel (byte 2 in BGRA) > 200
//   • That pixel's B channel (byte 0 in BGRA) is much lower than R
// This isolates the splatting shader from all upstream stages.
// ----------------------------------------------------------------
TEST_F(GaussianSplattingTest, splattingShaderIsolated) {
    const uint32_t W        = BackendConfig::WIDTH;   // 512
    const uint32_t H        = BackendConfig::HEIGHT;  // 384
    const uint32_t gridSize = 4;
    const uint32_t numBins  = gridSize * gridSize;    // 16

    // Gaussian centre on screen and which bin it falls in:
    //   cellW=128, cellH=96 → pixel (256,192) → gridX=2, gridY=2 → bin=10
    const float    gcx     = W * 0.5f;   // 256
    const float    gcy     = H * 0.5f;   // 192
    const uint32_t binIdx  = 10;

    // ---- Pre-computed Gaussian2D ----
    // The driver stores shader rgba8 (r,g,b,a) into physical BGRA as [b,g,r,a]:
    //   byte0=b_shader, byte1=g_shader, byte2=r_shader, byte3=a_shader
    // writePPM reads {p[2],p[1],p[0]} as {R,G,B} so:
    //   PPM R = byte2 = r_shader  →  write shader r=1 for PPM red
    //   PPM B = byte0 = b_shader  →  write shader b=0 to keep blue low
    //
    // covarianceInv = diag(1/400) → actual cov diag(400) → σ ≈ 20 px
    // Place gaussian at centre of bin 10 (x=[256,384), y=[192,288)) → (320,240)
    Gaussian2D g2d{};
    g2d.position   = {320.0f, 240.0f};            // centre of bin 10
    g2d.z          = 0.5f;
    g2d.binMask    = 1u << binIdx;
    g2d.covariance = glm::mat2(1.0f / 400.0f);   // stored as covarianceInv
    g2d.color      = {1.0f, 0.0f, 0.0f};          // shader r=1 → byte2=255 → PPM red
    g2d.alpha      = 1.0f;

    // ---- Pre-computed StartAndEnd table ----
    // 32 uint32_t = 16 × {start, end}, everything 0 except bin 10 end=1
    std::vector<uint32_t> boundsHost(numBins * 2, 0u);
    boundsHost[binIdx * 2 + 1] = 1u;   // end = 1

    // ---- Vulkan buffers (BufferElementSinglePath → allocated eagerly) ----
    auto bufSorted = std::make_shared<BufferElementSinglePath<Gaussian2DBuffer>>(*vulkanContext, 1);
    bufSorted->getBuffer().memcopyFrom(std::vector<Gaussian2D>{g2d});

    auto bufCount  = std::make_shared<BufferElementSinglePath<VulkanBuffer<uint32_t>>>(*vulkanContext, 1);
    const uint32_t one = 1u;
    bufCount->getBuffer(0).memcopyFrom(&one, 1);

    auto bufBounds = std::make_shared<BufferElementSinglePath<VulkanBuffer<uint32_t>>>(*vulkanContext, numBins * 2);
    bufBounds->getBuffer().memcopyFrom(boundsHost);

    // ---- ImageViewSrc from headless offscreen images ----
    uint32_t numImages = vulkanContext->getNumberOfSwapChainImages();
    VkExtent2D ext = vulkanContext->getSwapChainExtent();
    std::vector<VkImageView> ivs(numImages);
    std::vector<VkImage>     imgs(numImages);
    std::vector<VkExtent2D>  exts(numImages, ext);
    for (uint32_t i = 0; i < numImages; ++i) {
        ivs[i]  = vulkanContext->getImageView(i);
        imgs[i] = vulkanContext->getSwapChainImage(i);
    }
    auto imageViewSrc = std::make_shared<ImageViewSrc>(ivs, imgs, exts);

    // ---- Splatting stage ----
    const uint32_t tbX = 8, tbY = 8;
    const uint32_t gpbX = (W / tbX) / gridSize;   // 16
    const uint32_t gpbY = (H / tbY) / gridSize;   // 12

    auto splatStage = std::make_shared<GaussianSplatting>(
        *vulkanContext, "shaders/gsplat/gsplat_binned_splatting.comp.spv");
    splatStage->setInput(bufSorted,    0);
    splatStage->setInput(bufCount,     1);
    splatStage->setInput(bufBounds,    2);
    splatStage->setInput(imageViewSrc, 3);

    std::vector<SplatPushConstants> pcs;
    for (uint32_t y = 0; y < gridSize; ++y)
        for (uint32_t x = 0; x < gridSize; ++x)
            pcs.push_back({1u, gridSize, x, y, (float)W, (float)H});
    splatStage->setPushConstants(pcs);
    splatStage->setGroupCountX(gpbX);
    splatStage->setGroupCountY(gpbY);
    splatStage->setGroupCountZ(1);

    auto cg = ComputeGraph(*vulkanContext, 1);
    cg.compileFrom(splatStage);
    cg.submitAndWait(vulkanContext->getGraphicsQueue(), 0);

    // ---- Read back and verify ----
    saveImageAsPPM(*vulkanContext, imgs[0], W, H,
                   "test_splat_isolated.ppm");

    const VkDeviceSize bytes = W * H * 4;
    VkBuffer stBuf; VkDeviceMemory stMem;
    vulkanContext->createBuffer(bytes,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        stBuf, stMem);
    {
        VkCommandBufferAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool        = vulkanContext->getCommandPool();
        ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VkCommandBuffer cmd;
        vkAllocateCommandBuffers(vulkanContext->getDevice(), &ai, &cmd);
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &bi);
        VkBufferImageCopy region{};
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = {W, H, 1};
        vkCmdCopyImageToBuffer(cmd, imgs[0], VK_IMAGE_LAYOUT_GENERAL, stBuf, 1, &region);
        vkEndCommandBuffer(cmd);
        VkSubmitInfo si{}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
        vkQueueSubmit(vulkanContext->getGraphicsQueue(), 1, &si, VK_NULL_HANDLE);
        vkQueueWaitIdle(vulkanContext->getGraphicsQueue());
        vkFreeCommandBuffers(vulkanContext->getDevice(),
                             vulkanContext->getCommandPool(), 1, &cmd);
    }

    void* data;
    vkMapMemory(vulkanContext->getDevice(), stMem, 0, bytes, 0, &data);
    const uint8_t* px = static_cast<const uint8_t*>(data);

    // Check a 40×40 window around the gaussian centre (320, 240)
    const uint32_t cx = 320, cy = 240, r = 20;
    uint8_t maxByte2 = 0;   // byte2 = r_shader → PPM R  (should be bright)
    uint8_t maxByte0 = 0;   // byte0 = b_shader → PPM B  (should stay low)
    for (uint32_t y = cy - r; y <= cy + r; ++y) {
        for (uint32_t x = cx - r; x <= cx + r; ++x) {
            const uint8_t* p = px + (y * W + x) * 4;
            if (p[2] > maxByte2) maxByte2 = p[2];
            if (p[0] > maxByte0) maxByte0 = p[0];
        }
    }

    vkUnmapMemory(vulkanContext->getDevice(), stMem);
    vkFreeMemory(vulkanContext->getDevice(), stMem, nullptr);
    vkDestroyBuffer(vulkanContext->getDevice(), stBuf, nullptr);

    // Driver maps shader rgba8 (r,g,b,a) → physical BGRA bytes [b,g,r,a]:
    //   byte2 = r_shader = 1.0 → 255  (PPM red channel)
    //   byte0 = b_shader = 0.0 → 0    (PPM blue channel)
    EXPECT_GT(maxByte2, uint8_t(200))
        << "byte2 (r_shader → PPM red) should be bright near gaussian centre";
    EXPECT_LT(maxByte0, uint8_t(50))
        << "byte0 (b_shader → PPM blue) should be low";

    SUCCEED() << "Splatting shader isolation test: image written to test_splat_isolated.ppm";
}

// ----------------------------------------------------------------
// Test 8: VulkanGaussianSplatting — single frame via engine.step()
//
// Uses the production VulkanGaussianSplatting class exactly as the
// example app does, but runs exactly one frame through the headless
// engine and copies the result to a PPM for visual inspection.
//
// This is a divide-and-conquer test for the projection/synchronisation
// issues reported with the interactive example.  Running headless with
// submitAndWait-style synchronisation (vkQueueWaitIdle after step)
// eliminates double-buffering races and gives a stable, reproducible
// snapshot of what the full pipeline produces.
// ----------------------------------------------------------------
TEST_F(GaussianSplattingTest, vulkanGaussianSplattingSingleFrame) {
    const std::string spzPath = "3rdparty/spz/samples/racoonfamily.spz";
    if (!std::filesystem::exists(spzPath)) {
        GTEST_SKIP() << "SPZ sample not found: " << spzPath;
    }

    auto& engine = frontend->getKlartraumEngine();

    // ---- ImageViewSrc from headless offscreen images (with semaphores) ----
    uint32_t numImages = vulkanContext->getNumberOfSwapChainImages();
    VkExtent2D ext = vulkanContext->getSwapChainExtent();
    std::vector<VkImageView> ivs(numImages);
    std::vector<VkImage>     imgs(numImages);
    std::vector<VkExtent2D>  exts(numImages, ext);
    for (uint32_t i = 0; i < numImages; ++i) {
        ivs[i]  = vulkanContext->getImageView(i);
        imgs[i] = vulkanContext->getSwapChainImage(i);
    }
    auto imageViewSrc = std::make_shared<ImageViewSrc>(ivs, imgs, exts);
    for (uint32_t i = 0; i < numImages; ++i) {
        imageViewSrc->setWaitFor(i, vulkanContext->imageAvailableSemaphoresPerImage[i]);
    }

    // ---- Camera matching the example app defaults ----
    auto cameraUBO = std::make_shared<CameraUboType>();
    InterfaceCameraOrbit cameraOrbit(InterfaceCameraOrbit::UpDirection::Y);
    cameraOrbit.initialize(*vulkanContext);
    cameraOrbit.setAzimuth(0.9f);
    cameraOrbit.setElevation(-0.5f);
    cameraOrbit.setPosition({-0.5f, 0.0f, 0.5f});
    cameraOrbit.setDistance(1.0f);
    cameraOrbit.update(cameraUBO->ubo);

    // ---- Instantiate VulkanGaussianSplatting and add to engine ----
    auto splatting = vulkanContext->create<VulkanGaussianSplatting>(
        imageViewSrc, cameraUBO, spzPath);
    engine.add(splatting);

    // Upload camera matrices to GPU for all paths (after compileFrom)
    for (uint32_t i = 0; i < numImages; ++i) {
        cameraUBO->update(i);
    }

    // ---- Render exactly one frame, then drain the queue ----
    // engine.step() uses imageIndex = currentFrame % numImages = 0 for the
    // first call, so the result is in imgs[0].
    engine.step();
    vkQueueWaitIdle(vulkanContext->getGraphicsQueue());

    // ---- Copy image[0] (GENERAL layout, BGRA) to host and write PPM ----
    saveImageAsPPM(*vulkanContext, imgs[0],
                   ext.width, ext.height,
                   "test_gsplatting_single_frame.ppm");

    SUCCEED() << "Single frame written to test_gsplatting_single_frame.ppm ("
              << ext.width << "x" << ext.height << ")";
}
