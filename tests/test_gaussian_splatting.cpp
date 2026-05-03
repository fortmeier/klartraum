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

    // Chain: extractStage → sortOp → gatherStage → boundsStage → splatStage
    // We chain them manually via setInput; compile from the last element.
    // (Each is already connected to shared buffers; no additional setInput chaining
    //  is needed since they share buffer objects directly.)
    {
        auto cg2 = ComputeGraph(*vulkanContext, 1);
        cg2.compileFrom(splatStage);
        cg2.submitAndWait(vulkanContext->getGraphicsQueue(), 0);
    }

    // ---- Copy rendered image to host and write PPM ----
    saveImageAsPPM(*vulkanContext, imgs[0], (uint32_t)W, (uint32_t)H,
                   "test_gaussian_splatting_render.ppm");

    SUCCEED() << "Rendered " << (uint32_t)W << "x" << (uint32_t)H
              << " image written to test_gaussian_splatting_render.ppm";
}
