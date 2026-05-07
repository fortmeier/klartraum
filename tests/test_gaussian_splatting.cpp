/**
 * TESTS:
 * - projection: Projection shader with 3 gaussians, verifies screen-space positions/depths
 * - fullPipeline: Full 9-stage pipeline with a single gaussian, verifies binning count > 0 and sorted order
 * - singleRedGaussian: Full pipeline (manual stages) renders a red gaussian, checks pixel colors
 * - classWithSingleRedGaussian: VulkanGaussianSplatting class renders a single red gaussian, checks pixel colors
 * - classWithRaccoonScene: VulkanGaussianSplatting class renders raccoon SPZ scene and profiles GPU timing
 * - classWithRaccoonTwoFrames: VulkanGaussianSplatting class renders 4 frames, checks bit-exact determinism and bin coverage
 **/

#include <gtest/gtest.h>
#include <algorithm>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <random>
#include <vector>

#include <glm/glm.hpp>

#include "klartraum/headless_frontend.hpp"
#include "klartraum/vulkan_context.hpp"
#include "klartraum/vulkan_gaussian_splatting.hpp"
#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/generalcomputation.hpp"
#include "klartraum/computegraph/bufferelement.hpp"
#include "klartraum/vulkan_buffer.hpp"
#include "klartraum/vulkan_gaussian_splatting_types.hpp"
#include "klartraum/interface_camera_orbit.hpp"
#include "klartraum/backend_config.hpp"
#include "klartraum/computegraph/imageviewsrc.hpp"

using namespace klartraum;

class GaussianSplattingTest : public ::testing::Test {
protected:
    void SetUp() override {
        frontend = std::make_unique<HeadlessFrontend>();
        vulkanContext = &frontend->getKlartraumEngine().getVulkanContext();
    }
    void TearDown() override { frontend.reset(); }

    void makeCameraUBO(std::shared_ptr<CameraUboType>& out, float distance = 5.0f) {
        out = std::make_shared<CameraUboType>();
        InterfaceCameraOrbit orbit(InterfaceCameraOrbit::UpDirection::Y);
        orbit.initialize(*vulkanContext);
        orbit.setDistance(distance);
        orbit.update(out->ubo);
    }

    std::unique_ptr<HeadlessFrontend> frontend;
    VulkanContext* vulkanContext = nullptr;
};

// Helper: split Gaussian3D vector into per-field arrays for GPU upload.
// SH layout: band-major — shR[band * N + gaussianIdx]
struct GaussianInput3D {
    std::vector<glm::vec3> positions;
    std::vector<glm::vec4> rotations;
    std::vector<glm::vec3> scales;
    std::vector<glm::vec4> colorsAlpha;  // (r,g,b,alpha)
    std::vector<float>     shR;          // [15 * N], band-major
    std::vector<float>     shG;
    std::vector<float>     shB;

    static GaussianInput3D from(const std::vector<Gaussian3D>& src) {
        size_t N = src.size();
        GaussianInput3D out;
        out.positions.resize(N);
        out.rotations.resize(N);
        out.scales.resize(N);
        out.colorsAlpha.resize(N);
        out.shR.resize(15 * N, 0.0f);
        out.shG.resize(15 * N, 0.0f);
        out.shB.resize(15 * N, 0.0f);

        for (size_t i = 0; i < N; i++) {
            const auto& g = src[i];
            out.positions[i]   = {g.position[0], g.position[1], g.position[2]};
            out.rotations[i]   = {g.rotation[0], g.rotation[1], g.rotation[2], g.rotation[3]};
            out.scales[i]      = {g.scale[0],    g.scale[1],    g.scale[2]};
            out.colorsAlpha[i] = {g.color[0],    g.color[1],    g.color[2],    g.alpha};
            for (int b = 0; b < 15; b++) {
                out.shR[b * N + i] = g.shR[b];
                out.shG[b * N + i] = g.shG[b];
                out.shB[b * N + i] = g.shB[b];
            }
        }
        return out;
    }
};

// ----------------------------------------------------------------
// Test: projection
// Three gaussians at origin, Y+1, Z+1. Verifies screen-space
// positions and depths from the projection shader.
// ----------------------------------------------------------------
TEST_F(GaussianSplattingTest, projection) {
    const uint32_t N = 3;

    std::vector<Gaussian3D> g3d(N);
    g3d[0].position = {0.f, 0.f, 0.f};
    g3d[0].scale    = {1.f, 1.f, 1.f};
    g3d[0].rotation = {0.f, 0.f, 0.f, 1.f};
    g3d[0].alpha    = 0.5f;
    g3d[1].position = {0.f, 1.f, 0.f};
    g3d[1].scale    = {1.f, 1.f, 1.f};
    g3d[1].rotation = {0.f, 0.f, 0.f, 1.f};
    g3d[1].alpha    = 0.5f;
    g3d[2].position = {0.f, 0.f, 1.f};
    g3d[2].scale    = {1.f, 1.f, 1.f};
    g3d[2].rotation = {0.f, 0.f, 0.f, 1.f};
    g3d[2].alpha    = 0.5f;

    auto inp = GaussianInput3D::from(g3d);

    auto bufPos      = std::make_shared<BufferElement<VulkanBuffer<glm::vec3>>>(*vulkanContext, N);
    auto bufRot      = std::make_shared<BufferElement<VulkanBuffer<glm::vec4>>>(*vulkanContext, N);
    auto bufScale    = std::make_shared<BufferElement<VulkanBuffer<glm::vec3>>>(*vulkanContext, N);
    auto bufColAlpha = std::make_shared<BufferElement<VulkanBuffer<glm::vec4>>>(*vulkanContext, N);
    auto bufShR      = std::make_shared<BufferElement<VulkanBuffer<float>>>(*vulkanContext, 15 * N);
    auto bufShG      = std::make_shared<BufferElement<VulkanBuffer<float>>>(*vulkanContext, 15 * N);
    auto bufShB      = std::make_shared<BufferElement<VulkanBuffer<float>>>(*vulkanContext, 15 * N);

    std::shared_ptr<CameraUboType> cameraUBO;
    makeCameraUBO(cameraUBO);

    auto bufPos2D      = std::make_shared<BufferElement<VulkanBuffer<glm::vec2>>>(*vulkanContext, N);
    auto bufZ          = std::make_shared<BufferElement<VulkanBuffer<float>>>(*vulkanContext, N);
    auto bufBinMask    = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(*vulkanContext, N);
    auto bufCovInv     = std::make_shared<BufferElement<VulkanBuffer<glm::vec4>>>(*vulkanContext, N);
    auto bufColorAlpha = std::make_shared<BufferElement<VulkanBuffer<glm::vec4>>>(*vulkanContext, N);

    auto proj = std::make_shared<GaussianProjection>(
        *vulkanContext, "shaders/gsplat/gsplat_projection.comp.spv");
    proj->setInput(bufPos,        0);
    proj->setInput(bufRot,        1);
    proj->setInput(bufScale,      2);
    proj->setInput(bufColAlpha,   3);
    proj->setInput(bufShR,        4);
    proj->setInput(bufShG,        5);
    proj->setInput(bufShB,        6);
    proj->setInput(cameraUBO,     7);
    proj->setInput(bufPos2D,      8);
    proj->setInput(bufZ,          9);
    proj->setInput(bufBinMask,    10);
    proj->setInput(bufCovInv,     11);
    proj->setInput(bufColorAlpha, 12);
    proj->setPushConstants({{N, 4,
        (float)BackendConfig::WIDTH, (float)BackendConfig::HEIGHT}});
    proj->setGroupCountX(N / 128 + 1);

    auto cg = ComputeGraph(*vulkanContext, 1);
    cg.enableProfiling();
    cg.compileFrom(proj);

    bufPos->getBuffer(0).memcopyFrom(inp.positions);
    bufRot->getBuffer(0).memcopyFrom(inp.rotations);
    bufScale->getBuffer(0).memcopyFrom(inp.scales);
    bufColAlpha->getBuffer(0).memcopyFrom(inp.colorsAlpha);
    bufShR->getBuffer(0).memcopyFrom(inp.shR);
    bufShG->getBuffer(0).memcopyFrom(inp.shG);
    bufShB->getBuffer(0).memcopyFrom(inp.shB);
    cameraUBO->update(0);

    cg.submitAndWait(vulkanContext->getGraphicsQueue(), 0);

    std::vector<glm::vec2> outPos2D(N);
    std::vector<float>     outZ(N);
    bufPos2D->getBuffer(0).memcopyTo(outPos2D);
    bufZ->getBuffer(0).memcopyTo(outZ);

    const float cx = BackendConfig::WIDTH  * 0.5f;
    const float cy = BackendConfig::HEIGHT * 0.5f;

    // Origin → screen centre
    EXPECT_NEAR(outPos2D[0].x, cx, 3.0f);
    EXPECT_NEAR(outPos2D[0].y, cy, 3.0f);
    EXPECT_GT(outZ[0], 0.0f);
    EXPECT_LT(outZ[0], 1.0f);

    // Y+1 → same X, Y moves up (smaller Y, Vulkan-flipped)
    EXPECT_NEAR(outPos2D[1].x, cx, 3.0f);
    EXPECT_LT(outPos2D[1].y, cy);

    // Z+1 → X moves left (smaller X)
    EXPECT_LT(outPos2D[2].x, cx);
    EXPECT_NEAR(outPos2D[2].y, cy, 3.0f);

    auto results = cg.getProfilingResults();
    std::cout << "\n--- projection profiling ---\n";
    for (auto& [name, ms] : results)
        std::cout << "  " << name << ": " << ms << " ms\n";

    SUCCEED();
}

// ----------------------------------------------------------------
// Test: fullPipeline
// Full 9-stage pipeline: projection → binning (3-pass) → sort →
// gather → bounds → splatting. Single gaussian projected to bin 10.
// Verifies totalCount > 0 and sorted bin masks are non-decreasing.
// ----------------------------------------------------------------
TEST_F(GaussianSplattingTest, fullPipeline) {
    const float    W        = BackendConfig::WIDTH;
    const float    H        = BackendConfig::HEIGHT;
    const uint32_t gridSize = 4;
    const uint32_t numBins  = gridSize * gridSize;
    const uint32_t tpg      = 128;

    const uint32_t N = 1;
    Gaussian3D g{};
    g.position = {0.0f, -0.690f, -0.518f};
    g.scale    = {0.05f, 0.05f, 0.05f};
    g.rotation = {0.0f, 0.0f, 0.0f, 1.0f};
    g.color    = {1.772f, -1.772f, -1.772f};
    g.alpha    = 1.0f;
    auto inp = GaussianInput3D::from(std::vector<Gaussian3D>{g});

    const uint32_t numBinWGs  = N / tpg + 1;
    const uint32_t maxBinned  = N * numBins;
    const uint32_t numSortWGs = std::max(1u, std::min(320u, maxBinned / tpg + 1));

    std::shared_ptr<CameraUboType> cameraUBO;
    makeCameraUBO(cameraUBO);

    auto bufPos      = std::make_shared<BufferElement<VulkanBuffer<glm::vec3>>>(*vulkanContext, N);
    auto bufRot      = std::make_shared<BufferElement<VulkanBuffer<glm::vec4>>>(*vulkanContext, N);
    auto bufScale    = std::make_shared<BufferElement<VulkanBuffer<glm::vec3>>>(*vulkanContext, N);
    auto bufColAlpha = std::make_shared<BufferElement<VulkanBuffer<glm::vec4>>>(*vulkanContext, N);
    auto bufShR      = std::make_shared<BufferElement<VulkanBuffer<float>>>(*vulkanContext, 15*N);
    auto bufShG      = std::make_shared<BufferElement<VulkanBuffer<float>>>(*vulkanContext, 15*N);
    auto bufShB      = std::make_shared<BufferElement<VulkanBuffer<float>>>(*vulkanContext, 15*N);

    auto bufProjPos2D   = std::make_shared<BufferElement<VulkanBuffer<glm::vec2>>>(*vulkanContext, N);
    auto bufProjZ       = std::make_shared<BufferElement<VulkanBuffer<float>>>(*vulkanContext, N);
    auto bufProjBinMask = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(*vulkanContext, N);
    auto bufProjCovInv  = std::make_shared<BufferElement<VulkanBuffer<glm::vec4>>>(*vulkanContext, N);
    auto bufProjCA      = std::make_shared<BufferElement<VulkanBuffer<glm::vec4>>>(*vulkanContext, N);

    auto bufHistogram  = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(*vulkanContext, numBins * numBinWGs);
    bufHistogram->setRecordToZero(true);
    auto bufOffsets    = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(*vulkanContext, numBinWGs + 1);
    bufOffsets->setRecordToZero(true);
    auto bufTotalCount = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(*vulkanContext, 1);
    bufTotalCount->setRecordToZero(true);

    auto bufBinPos2D  = std::make_shared<BufferElement<VulkanBuffer<glm::vec2>>>(*vulkanContext, maxBinned);
    auto bufBinZ      = std::make_shared<BufferElement<VulkanBuffer<float>>>(*vulkanContext, maxBinned);
    auto bufBinMask   = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(*vulkanContext, maxBinned);
    auto bufBinCovInv = std::make_shared<BufferElement<VulkanBuffer<glm::vec4>>>(*vulkanContext, maxBinned);
    auto bufBinCA     = std::make_shared<BufferElement<VulkanBuffer<glm::vec4>>>(*vulkanContext, maxBinned);

    auto bufValA = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(*vulkanContext, maxBinned);
    auto bufIdxA = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(*vulkanContext, maxBinned);
    auto bufValB = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(*vulkanContext, maxBinned);
    auto bufIdxB = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(*vulkanContext, maxBinned);
    auto scratchHist    = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(*vulkanContext, numBins * numSortWGs);
    scratchHist->setRecordToZero(true);
    auto scratchCounts  = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(*vulkanContext, numBins);
    scratchCounts->setRecordToZero(true);
    auto scratchOffsets = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(*vulkanContext, numSortWGs + 1);
    scratchOffsets->setRecordToZero(true);

    auto bufSortPos2D  = std::make_shared<BufferElement<VulkanBuffer<glm::vec2>>>(*vulkanContext, maxBinned);
    auto bufSortZ      = std::make_shared<BufferElement<VulkanBuffer<float>>>(*vulkanContext, maxBinned);
    auto bufSortMask   = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(*vulkanContext, maxBinned);
    auto bufSortCovInv = std::make_shared<BufferElement<VulkanBuffer<glm::vec4>>>(*vulkanContext, maxBinned);
    auto bufSortCA     = std::make_shared<BufferElement<VulkanBuffer<glm::vec4>>>(*vulkanContext, maxBinned);

    auto bufBounds = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(*vulkanContext, numBins * 2);
    bufBounds->setRecordToZero(true);

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

    auto submitStage = [&](auto stage) {
        auto cg = ComputeGraph(*vulkanContext, 1);
        cg.compileFrom(stage);
        cg.submitAndWait(vulkanContext->getGraphicsQueue(), 0);
    };

    // Stage 1: projection
    {
        auto proj = std::make_shared<GaussianProjection>(
            *vulkanContext, "shaders/gsplat/gsplat_projection.comp.spv");
        proj->setInput(bufPos,       0); proj->setInput(bufRot,        1);
        proj->setInput(bufScale,     2); proj->setInput(bufColAlpha,   3);
        proj->setInput(bufShR,       4); proj->setInput(bufShG,        5);
        proj->setInput(bufShB,       6); proj->setInput(cameraUBO,     7);
        proj->setInput(bufProjPos2D, 8); proj->setInput(bufProjZ,      9);
        proj->setInput(bufProjBinMask, 10); proj->setInput(bufProjCovInv, 11);
        proj->setInput(bufProjCA,    12);
        proj->setPushConstants({{N, gridSize, W, H}});
        proj->setGroupCountX(N / tpg + 1);
        auto cg = ComputeGraph(*vulkanContext, 1);
        cg.compileFrom(proj);
        bufPos->getBuffer(0).memcopyFrom(inp.positions);
        bufRot->getBuffer(0).memcopyFrom(inp.rotations);
        bufScale->getBuffer(0).memcopyFrom(inp.scales);
        bufColAlpha->getBuffer(0).memcopyFrom(inp.colorsAlpha);
        bufShR->getBuffer(0).memcopyFrom(inp.shR);
        bufShG->getBuffer(0).memcopyFrom(inp.shG);
        bufShB->getBuffer(0).memcopyFrom(inp.shB);
        cameraUBO->update(0);
        cg.submitAndWait(vulkanContext->getGraphicsQueue(), 0);
    }

    // Stage 2: binning count
    {
        auto count = std::make_shared<GaussianBinningCount>(
            *vulkanContext, "shaders/gsplat/gsplat_binning_count.comp.spv");
        count->setInput(bufProjPos2D, 0); count->setInput(bufProjZ,      1);
        count->setInput(bufProjCovInv, 2); count->setInput(bufHistogram, 3);
        count->setInput(bufOffsets,   4);
        count->setPushConstants({{N, gridSize, W, H}});
        count->setGroupCountX(numBinWGs);
        submitStage(count);
        bufHistogram->setRecordToZero(false);
        bufOffsets->setRecordToZero(false);
    }

    // Stage 3: binning prefix sum
    {
        auto ps = std::make_shared<GeneralComputation<>>(
            *vulkanContext, "shaders/gsplat/gsplat_binning_prefix_sum.comp.spv");
        ps->setInput(bufHistogram, 0); ps->setInput(bufOffsets, 1);
        ps->setGroupCountX(numBinWGs);
        submitStage(ps);
    }

    // Stage 4: binning scatter
    {
        auto scatter = std::make_shared<GaussianBinningScatter>(
            *vulkanContext, "shaders/gsplat/gsplat_binning_scatter.comp.spv");
        scatter->setInput(bufProjPos2D,  0); scatter->setInput(bufProjZ,      1);
        scatter->setInput(bufProjCovInv, 2); scatter->setInput(bufProjCA,     3);
        scatter->setInput(bufHistogram,  4); scatter->setInput(bufTotalCount, 5);
        scatter->setInput(bufBinPos2D,   6); scatter->setInput(bufBinZ,       7);
        scatter->setInput(bufBinMask,    8); scatter->setInput(bufBinCovInv,  9);
        scatter->setInput(bufBinCA,      10);
        scatter->setPushConstants({{N, gridSize, W, H, maxBinned}});
        scatter->setGroupCountX(numBinWGs);
        submitStage(scatter);
        bufTotalCount->setRecordToZero(false);
    }

    uint32_t totalCount = 0;
    { std::vector<uint32_t> tmp(1); bufTotalCount->getBuffer(0).memcopyTo(tmp); totalCount = tmp[0]; }
    ASSERT_GT(totalCount, 0u) << "Binning produced zero entries";
    std::cout << "  totalCount after binning: " << totalCount << "\n";

    // Stages 5+6: extract sort keys → radix sort
    auto sortOp = std::make_shared<RadixSort>(*vulkanContext, std::vector<std::string>{
        "shaders/gsplat/gsplat_radix_sort_histogram.comp.spv",
        "shaders/gsplat/gsplat_radix_sort_hist_prefix_sum.comp.spv",
        "shaders/gsplat/gsplat_radix_sort_hist_scatter.comp.spv"
    });
    sortOp->setInput(bufValA, 0); sortOp->setInput(bufIdxA, 1);
    sortOp->setInput(bufValB, 2); sortOp->setInput(bufIdxB, 3);
    sortOp->addScratchBufferElement(scratchCounts,  true);
    sortOp->addScratchBufferElement(scratchOffsets, true);
    sortOp->addScratchBufferElement(bufTotalCount,  false);
    sortOp->addScratchBufferElement(scratchHist,    true);
    sortOp->setGroupCountX(numSortWGs);
    { std::vector<SortPushConstants> pcs;
      for (uint32_t i = 0; i < 32 / 4; ++i) pcs.push_back({i, totalCount, numBins});
      sortOp->setPushConstants(pcs); }

    ComputeGraph cgSort(*vulkanContext, 1);
    cgSort.compileFrom(sortOp);

    { std::vector<uint32_t> sentinel(maxBinned, 0xFFFFFFFFu);
      bufValA->getBuffer(0).memcopyFrom(sentinel);
      bufIdxA->getBuffer(0).memcopyFrom(sentinel); }

    { auto extract = std::make_shared<GeneralComputation<>>(
          *vulkanContext, "shaders/gsplat/gsplat_extract_sort_keys.comp.spv");
      extract->setInput(bufBinMask,    0); extract->setInput(bufBinZ,       1);
      extract->setInput(bufTotalCount, 2); extract->setInput(bufValA,       3);
      extract->setInput(bufIdxA,       4);
      extract->setGroupCountX(maxBinned / tpg + 1);
      submitStage(extract); }

    cgSort.submitAndWait(vulkanContext->getGraphicsQueue(), 0);

    // Stage 7: gather sorted
    { auto gather = std::make_shared<GeneralComputation<>>(
          *vulkanContext, "shaders/gsplat/gsplat_gather_sorted.comp.spv");
      gather->setInput(bufBinPos2D,  0); gather->setInput(bufBinZ,       1);
      gather->setInput(bufBinMask,   2); gather->setInput(bufBinCovInv,  3);
      gather->setInput(bufBinCA,     4); gather->setInput(bufIdxA,       5);
      gather->setInput(bufTotalCount,6); gather->setInput(bufSortPos2D,  7);
      gather->setInput(bufSortZ,     8); gather->setInput(bufSortMask,   9);
      gather->setInput(bufSortCovInv,10); gather->setInput(bufSortCA,   11);
      gather->setGroupCountX(maxBinned / tpg + 1);
      submitStage(gather); }

    // Stage 8: bin bounds
    { auto bounds = std::make_shared<GeneralComputation<>>(
          *vulkanContext, "shaders/gsplat/gsplat_bin_bounds.comp.spv");
      bounds->setInput(bufSortMask,   0); bounds->setInput(bufTotalCount, 1);
      bounds->setInput(bufBounds,     2);
      bounds->setGroupCountX(maxBinned / 256 + 1);
      submitStage(bounds);
      bufBounds->setRecordToZero(false); }

    // Stage 9: splatting
    { const uint32_t tbX = 8, tbY = 8;
      const uint32_t gpbX = uint32_t((W / tbX) / gridSize);
      const uint32_t gpbY = uint32_t((H / tbY) / gridSize);
      auto splat = std::make_shared<GaussianSplatting>(
          *vulkanContext, "shaders/gsplat/gsplat_binned_splatting.comp.spv");
      splat->setInput(bufSortPos2D,  0); splat->setInput(bufSortCovInv, 1);
      splat->setInput(bufSortCA,     2); splat->setInput(bufTotalCount, 3);
      splat->setInput(bufBounds,     4); splat->setInput(imageViewSrc,  5);
      std::vector<SplatPushConstants> pcs;
      for (uint32_t y = 0; y < gridSize; ++y)
          for (uint32_t x = 0; x < gridSize; ++x)
              pcs.push_back({totalCount, gridSize, x, y, W, H});
      splat->setPushConstants(pcs);
      splat->setGroupCountX(gpbX); splat->setGroupCountY(gpbY); splat->setGroupCountZ(1);
      submitStage(splat); }

    std::vector<uint32_t> sortedMasks(totalCount);
    bufSortMask->getBuffer(0).memcopyTo(sortedMasks);
    for (uint32_t i = 1; i < totalCount; ++i)
        EXPECT_LE(sortedMasks[i-1], sortedMasks[i]) << "Bin masks not sorted at " << i;

    std::cout << "  fullPipeline: OK, " << totalCount << " binned gaussians rendered\n";
    SUCCEED();
}

// Helper: copy a rendered swapchain image to host memory.
static std::vector<uint8_t> readImage(VulkanContext& vc, VkImage image,
                                      uint32_t W, uint32_t H)
{
    const VkDeviceSize bytes = W * H * 4;
    VkBuffer buf; VkDeviceMemory mem;
    vc.createBuffer(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    buf, mem);
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
    region.imageExtent = {W, H, 1};
    vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_GENERAL, buf, 1, &region);
    vkEndCommandBuffer(cmd);
    VkSubmitInfo si{}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    vkQueueSubmit(vc.getGraphicsQueue(), 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(vc.getGraphicsQueue());
    vkFreeCommandBuffers(vc.getDevice(), vc.getCommandPool(), 1, &cmd);
    void* data;
    vkMapMemory(vc.getDevice(), mem, 0, bytes, 0, &data);
    std::vector<uint8_t> pixels(static_cast<uint8_t*>(data),
                                 static_cast<uint8_t*>(data) + bytes);
    vkUnmapMemory(vc.getDevice(), mem);
    vkDestroyBuffer(vc.getDevice(), buf, nullptr);
    vkFreeMemory(vc.getDevice(), mem, nullptr);
    return pixels;
}

// Helper: run the complete pipeline for a given Gaussian3D set; return the rendered image.
static std::vector<uint8_t> render(VulkanContext& vc,
                                   const std::vector<Gaussian3D>& g3dVec,
                                   std::shared_ptr<CameraUboType> cameraUBO)
{
    const float    W        = BackendConfig::WIDTH;
    const float    H        = BackendConfig::HEIGHT;
    const uint32_t gridSize = 4;
    const uint32_t numBins  = gridSize * gridSize;
    const uint32_t tpg      = 128;
    const uint32_t N        = (uint32_t)g3dVec.size();

    auto inp = GaussianInput3D::from(g3dVec);

    const uint32_t numBinWGs  = N / tpg + 1;
    const uint32_t maxBinned  = N * numBins;
    const uint32_t numSortWGs = std::max(1u, std::min(320u, maxBinned / tpg + 1));

    auto bufPos      = std::make_shared<BufferElement<VulkanBuffer<glm::vec3>>>(vc, N);
    auto bufRot      = std::make_shared<BufferElement<VulkanBuffer<glm::vec4>>>(vc, N);
    auto bufScale    = std::make_shared<BufferElement<VulkanBuffer<glm::vec3>>>(vc, N);
    auto bufColAlpha = std::make_shared<BufferElement<VulkanBuffer<glm::vec4>>>(vc, N);
    auto bufShR      = std::make_shared<BufferElement<VulkanBuffer<float>>>(vc, 15*N);
    auto bufShG      = std::make_shared<BufferElement<VulkanBuffer<float>>>(vc, 15*N);
    auto bufShB      = std::make_shared<BufferElement<VulkanBuffer<float>>>(vc, 15*N);

    auto bufProjPos2D   = std::make_shared<BufferElement<VulkanBuffer<glm::vec2>>>(vc, N);
    auto bufProjZ       = std::make_shared<BufferElement<VulkanBuffer<float>>>(vc, N);
    auto bufProjBinMask = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vc, N);
    auto bufProjCovInv  = std::make_shared<BufferElement<VulkanBuffer<glm::vec4>>>(vc, N);
    auto bufProjCA      = std::make_shared<BufferElement<VulkanBuffer<glm::vec4>>>(vc, N);

    auto bufHistogram  = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vc, numBins * numBinWGs);
    bufHistogram->setRecordToZero(true);
    auto bufOffsets    = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vc, numBinWGs + 1);
    bufOffsets->setRecordToZero(true);
    auto bufTotalCount = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vc, 1);
    bufTotalCount->setRecordToZero(true);

    auto bufBinPos2D  = std::make_shared<BufferElement<VulkanBuffer<glm::vec2>>>(vc, maxBinned);
    auto bufBinZ      = std::make_shared<BufferElement<VulkanBuffer<float>>>(vc, maxBinned);
    auto bufBinMask   = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vc, maxBinned);
    auto bufBinCovInv = std::make_shared<BufferElement<VulkanBuffer<glm::vec4>>>(vc, maxBinned);
    auto bufBinCA     = std::make_shared<BufferElement<VulkanBuffer<glm::vec4>>>(vc, maxBinned);

    auto bufValA = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vc, maxBinned);
    auto bufIdxA = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vc, maxBinned);
    auto bufValB = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vc, maxBinned);
    auto bufIdxB = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vc, maxBinned);
    auto scratchHist    = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vc, numBins * numSortWGs);
    scratchHist->setRecordToZero(true);
    auto scratchCounts  = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vc, numBins);
    scratchCounts->setRecordToZero(true);
    auto scratchOffsets = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vc, numSortWGs + 1);
    scratchOffsets->setRecordToZero(true);

    auto bufSortPos2D  = std::make_shared<BufferElement<VulkanBuffer<glm::vec2>>>(vc, maxBinned);
    auto bufSortZ      = std::make_shared<BufferElement<VulkanBuffer<float>>>(vc, maxBinned);
    auto bufSortMask   = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vc, maxBinned);
    auto bufSortCovInv = std::make_shared<BufferElement<VulkanBuffer<glm::vec4>>>(vc, maxBinned);
    auto bufSortCA     = std::make_shared<BufferElement<VulkanBuffer<glm::vec4>>>(vc, maxBinned);
    auto bufBounds     = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vc, numBins * 2);
    bufBounds->setRecordToZero(true);

    uint32_t numImages = vc.getNumberOfSwapChainImages();
    VkExtent2D ext = vc.getSwapChainExtent();
    std::vector<VkImageView> ivs(numImages);
    std::vector<VkImage>     imgs(numImages);
    std::vector<VkExtent2D>  exts(numImages, ext);
    for (uint32_t i = 0; i < numImages; ++i) { ivs[i] = vc.getImageView(i); imgs[i] = vc.getSwapChainImage(i); }
    auto imageViewSrc = std::make_shared<ImageViewSrc>(ivs, imgs, exts);

    auto submitS = [&](auto stage) {
        auto cg = ComputeGraph(vc, 1);
        cg.compileFrom(stage);
        cg.submitAndWait(vc.getGraphicsQueue(), 0);
    };

    // Projection
    { auto proj = std::make_shared<GaussianProjection>(vc, "shaders/gsplat/gsplat_projection.comp.spv");
      proj->setInput(bufPos, 0); proj->setInput(bufRot, 1); proj->setInput(bufScale, 2);
      proj->setInput(bufColAlpha, 3); proj->setInput(bufShR, 4); proj->setInput(bufShG, 5);
      proj->setInput(bufShB, 6); proj->setInput(cameraUBO, 7);
      proj->setInput(bufProjPos2D, 8); proj->setInput(bufProjZ, 9); proj->setInput(bufProjBinMask, 10);
      proj->setInput(bufProjCovInv, 11); proj->setInput(bufProjCA, 12);
      proj->setPushConstants({{N, gridSize, W, H}});
      proj->setGroupCountX(N / tpg + 1);
      auto cg = ComputeGraph(vc, 1); cg.compileFrom(proj);
      bufPos->getBuffer(0).memcopyFrom(inp.positions); bufRot->getBuffer(0).memcopyFrom(inp.rotations);
      bufScale->getBuffer(0).memcopyFrom(inp.scales);  bufColAlpha->getBuffer(0).memcopyFrom(inp.colorsAlpha);
      bufShR->getBuffer(0).memcopyFrom(inp.shR); bufShG->getBuffer(0).memcopyFrom(inp.shG);
      bufShB->getBuffer(0).memcopyFrom(inp.shB); cameraUBO->update(0);
      cg.submitAndWait(vc.getGraphicsQueue(), 0); }

    // Count
    { auto c = std::make_shared<GaussianBinningCount>(vc, "shaders/gsplat/gsplat_binning_count.comp.spv");
      c->setInput(bufProjPos2D, 0); c->setInput(bufProjZ, 1); c->setInput(bufProjCovInv, 2);
      c->setInput(bufHistogram, 3); c->setInput(bufOffsets, 4);
      c->setPushConstants({{N, gridSize, W, H}}); c->setGroupCountX(numBinWGs); submitS(c);
      bufHistogram->setRecordToZero(false); bufOffsets->setRecordToZero(false); }

    // PrefixSum
    { auto ps = std::make_shared<GeneralComputation<>>(vc, "shaders/gsplat/gsplat_binning_prefix_sum.comp.spv");
      ps->setInput(bufHistogram, 0); ps->setInput(bufOffsets, 1); ps->setGroupCountX(numBinWGs); submitS(ps); }

    // Scatter
    { auto sc = std::make_shared<GaussianBinningScatter>(vc, "shaders/gsplat/gsplat_binning_scatter.comp.spv");
      sc->setInput(bufProjPos2D, 0); sc->setInput(bufProjZ, 1); sc->setInput(bufProjCovInv, 2);
      sc->setInput(bufProjCA, 3); sc->setInput(bufHistogram, 4); sc->setInput(bufTotalCount, 5);
      sc->setInput(bufBinPos2D, 6); sc->setInput(bufBinZ, 7); sc->setInput(bufBinMask, 8);
      sc->setInput(bufBinCovInv, 9); sc->setInput(bufBinCA, 10);
      sc->setPushConstants({{N, gridSize, W, H, maxBinned}}); sc->setGroupCountX(numBinWGs); submitS(sc);
      bufTotalCount->setRecordToZero(false); }

    uint32_t totalCount = 0;
    { std::vector<uint32_t> tmp(1); bufTotalCount->getBuffer(0).memcopyTo(tmp); totalCount = tmp[0]; }
    if (totalCount == 0) return {};

    // Sort
    auto sortOp = std::make_shared<RadixSort>(vc, std::vector<std::string>{
        "shaders/gsplat/gsplat_radix_sort_histogram.comp.spv",
        "shaders/gsplat/gsplat_radix_sort_hist_prefix_sum.comp.spv",
        "shaders/gsplat/gsplat_radix_sort_hist_scatter.comp.spv"});
    sortOp->setInput(bufValA, 0); sortOp->setInput(bufIdxA, 1);
    sortOp->setInput(bufValB, 2); sortOp->setInput(bufIdxB, 3);
    sortOp->addScratchBufferElement(scratchCounts, true); sortOp->addScratchBufferElement(scratchOffsets, true);
    sortOp->addScratchBufferElement(bufTotalCount, false); sortOp->addScratchBufferElement(scratchHist, true);
    sortOp->setGroupCountX(numSortWGs);
    { std::vector<SortPushConstants> pcs;
      for (uint32_t i = 0; i < 8; ++i) pcs.push_back({i, maxBinned, numBins});
      sortOp->setPushConstants(pcs); }
    ComputeGraph cgSort(vc, 1);
    cgSort.compileFrom(sortOp);
    { std::vector<uint32_t> sent(maxBinned, 0xFFFFFFFFu);
      bufValA->getBuffer(0).memcopyFrom(sent); bufIdxA->getBuffer(0).memcopyFrom(sent); }
    { auto ex = std::make_shared<GeneralComputation<>>(vc, "shaders/gsplat/gsplat_extract_sort_keys.comp.spv");
      ex->setInput(bufBinMask, 0); ex->setInput(bufBinZ, 1); ex->setInput(bufTotalCount, 2);
      ex->setInput(bufValA, 3); ex->setInput(bufIdxA, 4);
      ex->setGroupCountX(maxBinned / tpg + 1); submitS(ex); }
    cgSort.submitAndWait(vc.getGraphicsQueue(), 0);

    // Gather + bounds + splat
    { auto g = std::make_shared<GeneralComputation<>>(vc, "shaders/gsplat/gsplat_gather_sorted.comp.spv");
      g->setInput(bufBinPos2D, 0); g->setInput(bufBinZ, 1); g->setInput(bufBinMask, 2);
      g->setInput(bufBinCovInv, 3); g->setInput(bufBinCA, 4); g->setInput(bufIdxA, 5);
      g->setInput(bufTotalCount, 6); g->setInput(bufSortPos2D, 7); g->setInput(bufSortZ, 8);
      g->setInput(bufSortMask, 9); g->setInput(bufSortCovInv, 10); g->setInput(bufSortCA, 11);
      g->setGroupCountX(maxBinned / tpg + 1); submitS(g); }
    { auto b = std::make_shared<GeneralComputation<>>(vc, "shaders/gsplat/gsplat_bin_bounds.comp.spv");
      b->setInput(bufSortMask, 0); b->setInput(bufTotalCount, 1); b->setInput(bufBounds, 2);
      b->setGroupCountX(maxBinned / 256 + 1); submitS(b); bufBounds->setRecordToZero(false); }
    { const uint32_t tbX=8, tbY=8, gpbX=uint32_t((W/tbX)/gridSize), gpbY=uint32_t((H/tbY)/gridSize);
      auto sp = std::make_shared<GaussianSplatting>(vc, "shaders/gsplat/gsplat_binned_splatting.comp.spv");
      sp->setInput(bufSortPos2D, 0); sp->setInput(bufSortCovInv, 1); sp->setInput(bufSortCA, 2);
      sp->setInput(bufTotalCount, 3); sp->setInput(bufBounds, 4); sp->setInput(imageViewSrc, 5);
      std::vector<SplatPushConstants> pcs;
      for (uint32_t y = 0; y < gridSize; ++y)
          for (uint32_t x = 0; x < gridSize; ++x)
              pcs.push_back({totalCount, gridSize, x, y, W, H});
      sp->setPushConstants(pcs); sp->setGroupCountX(gpbX); sp->setGroupCountY(gpbY); sp->setGroupCountZ(1);
      submitS(sp); }

    return readImage(vc, imgs[0], (uint32_t)W, (uint32_t)H);
}

// ----------------------------------------------------------------
// Test: singleRedGaussian
// Renders a single gaussian with SH coefficients that map to red.
// The gaussian projects to screen centre (320,240) = bin 10.
// Verifies byte2 (red channel) > 200 and byte0 (blue) < 50 in a
// 60x60 window around the expected position.
// ----------------------------------------------------------------
TEST_F(GaussianSplattingTest, singleRedGaussian) {
    const float W = BackendConfig::WIDTH;
    const float H = BackendConfig::HEIGHT;

    Gaussian3D g{};
    g.position = {0.0f, -0.690f, -0.518f};
    g.scale    = {0.05f, 0.05f, 0.05f};
    g.rotation = {0.0f, 0.0f, 0.0f, 1.0f};
    g.color    = {1.772f, -1.772f, -1.772f};
    g.alpha    = 1.0f;

    std::shared_ptr<CameraUboType> cameraUBO;
    makeCameraUBO(cameraUBO);

    auto pixels = render(*vulkanContext, {g}, cameraUBO);
    ASSERT_FALSE(pixels.empty()) << "Pipeline produced zero binned gaussians";

    const int cx = 320, cy = 240, half = 30;
    uint8_t maxByte2 = 0, maxByte0 = 0;
    for (int y = cy - half; y <= cy + half; ++y) {
        for (int x = cx - half; x <= cx + half; ++x) {
            if (x < 0 || x >= (int)W || y < 0 || y >= (int)H) continue;
            size_t base = (y * (size_t)W + x) * 4;
            maxByte2 = std::max(maxByte2, pixels[base + 2]);
            maxByte0 = std::max(maxByte0, pixels[base + 0]);
        }
    }
    std::cout << "  singleRedGaussian: maxByte2(red)=" << (int)maxByte2
              << " maxByte0(blue)=" << (int)maxByte0 << "\n";
    EXPECT_GT(maxByte2, uint8_t(200)) << "red channel not bright near gaussian centre";
    EXPECT_LT(maxByte0, uint8_t(50))  << "blue channel should stay low";
}

// ----------------------------------------------------------------
// Helper: copy a VkImage (BGRA GENERAL) to a host vector.
// ----------------------------------------------------------------
static std::vector<uint8_t> readImageToHost(VulkanContext& vc, VkImage image,
                                             uint32_t W, uint32_t H)
{
    const VkDeviceSize bytes = W * H * 4;
    VkBuffer buf; VkDeviceMemory mem;
    vc.createBuffer(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    buf, mem);
    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = vc.getCommandPool(); ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
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
    VkSubmitInfo si{}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    vkQueueSubmit(vc.getGraphicsQueue(), 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(vc.getGraphicsQueue());
    vkFreeCommandBuffers(vc.getDevice(), vc.getCommandPool(), 1, &cmd);
    void* data;
    vkMapMemory(vc.getDevice(), mem, 0, bytes, 0, &data);
    std::vector<uint8_t> result(static_cast<const uint8_t*>(data),
                                static_cast<const uint8_t*>(data) + bytes);
    vkUnmapMemory(vc.getDevice(), mem);
    vkFreeMemory(vc.getDevice(), mem, nullptr);
    vkDestroyBuffer(vc.getDevice(), buf, nullptr);
    return result;
}

// Helper: write BGRA pixels to a binary PPM file (B↔R swapped for PPM RGB).
static void writePPM(const std::string& path, const uint8_t* bgra,
                     uint32_t W, uint32_t H)
{
    std::ofstream f(path, std::ios::binary);
    f << "P6\n" << W << " " << H << "\n255\n";
    for (uint32_t y = 0; y < H; ++y)
        for (uint32_t x = 0; x < W; ++x) {
            const uint8_t* p = bgra + (y * W + x) * 4;
            uint8_t rgb[3] = { p[2], p[1], p[0] };
            f.write(reinterpret_cast<const char*>(rgb), 3);
        }
}

// Helper: build an ImageViewSrc from the headless swapchain images.
static std::shared_ptr<ImageViewSrc> makeImageViewSrc(
    VulkanContext& vc,
    std::vector<VkImage>& outImgs)
{
    uint32_t numImages = vc.getNumberOfSwapChainImages();
    VkExtent2D ext = vc.getSwapChainExtent();
    std::vector<VkImageView> ivs(numImages);
    outImgs.resize(numImages);
    std::vector<VkExtent2D>  exts(numImages, ext);
    for (uint32_t i = 0; i < numImages; ++i) {
        ivs[i]      = vc.getImageView(i);
        outImgs[i]  = vc.getSwapChainImage(i);
    }
    return std::make_shared<ImageViewSrc>(ivs, outImgs, exts);
}

// ----------------------------------------------------------------
// Test: classWithSingleRedGaussian
// Uses VulkanGaussianSplatting class (not manual pipeline stages).
// Provides a single red gaussian via the vector constructor.
// Verifies the rendered pixel near (320,240) is bright red.
// ----------------------------------------------------------------
TEST_F(GaussianSplattingTest, classWithSingleRedGaussian) {
    auto& engine = frontend->getKlartraumEngine();

    std::vector<VkImage> imgs;
    auto imageViewSrc = makeImageViewSrc(*vulkanContext, imgs);
    for (uint32_t i = 0; i < vulkanContext->getNumberOfSwapChainImages(); ++i)
        imageViewSrc->setWaitFor(i, vulkanContext->imageAvailableSemaphoresPerImage[i]);

    auto cameraUBO = std::make_shared<CameraUboType>();
    InterfaceCameraOrbit orbit(InterfaceCameraOrbit::UpDirection::Y);
    orbit.initialize(*vulkanContext);
    orbit.setDistance(5.0f);
    orbit.update(cameraUBO->ubo);

    Gaussian3D g{};
    g.position = {0.0f, -0.690f, -0.518f};
    g.scale    = {0.05f, 0.05f, 0.05f};
    g.rotation = {0.0f, 0.0f, 0.0f, 1.0f};
    g.color    = {1.772f, -1.772f, -1.772f};
    g.alpha    = 1.0f;

    auto splatting = vulkanContext->create<VulkanGaussianSplatting>(
        imageViewSrc, cameraUBO, std::vector<Gaussian3D>{g});
    engine.add(splatting);

    for (uint32_t i = 0; i < vulkanContext->getNumberOfSwapChainImages(); ++i)
        cameraUBO->update(i);

    engine.step();
    vkQueueWaitIdle(vulkanContext->getGraphicsQueue());

    VkExtent2D ext = vulkanContext->getSwapChainExtent();
    auto pixels = readImageToHost(*vulkanContext, imgs[0], ext.width, ext.height);

    const int cx = 320, cy = 240, half = 30;
    uint8_t maxByte2 = 0, maxByte0 = 0;
    for (int y = cy - half; y <= cy + half; ++y)
        for (int x = cx - half; x <= cx + half; ++x) {
            if (x < 0 || x >= (int)ext.width || y < 0 || y >= (int)ext.height) continue;
            size_t base = (y * (size_t)ext.width + x) * 4;
            maxByte2 = std::max(maxByte2, pixels[base + 2]);
            maxByte0 = std::max(maxByte0, pixels[base + 0]);
        }
    std::cout << "  classWithSingleRedGaussian: maxByte2(red)=" << (int)maxByte2
              << " maxByte0(blue)=" << (int)maxByte0 << "\n";
    EXPECT_GT(maxByte2, uint8_t(200)) << "red channel not bright near gaussian centre";
    EXPECT_LT(maxByte0, uint8_t(50))  << "blue channel should stay low";
}

// ----------------------------------------------------------------
// Test: classWithRaccoonScene
// Uses VulkanGaussianSplatting class with the raccoon SPZ file.
// Renders several frames, reports per-stage GPU profiling.
// Verifies the image is not all-black (pipeline actually drew something).
// ----------------------------------------------------------------
TEST_F(GaussianSplattingTest, classWithRaccoonScene) {
    const std::string spzPath = "3rdparty/spz/samples/racoonfamily.spz";
    if (!std::filesystem::exists(spzPath)) {
        GTEST_SKIP() << "SPZ sample not found: " << spzPath;
    }

    auto& engine = frontend->getKlartraumEngine();
    engine.enableProfiling();

    std::vector<VkImage> imgs;
    auto imageViewSrc = makeImageViewSrc(*vulkanContext, imgs);
    for (uint32_t i = 0; i < vulkanContext->getNumberOfSwapChainImages(); ++i)
        imageViewSrc->setWaitFor(i, vulkanContext->imageAvailableSemaphoresPerImage[i]);

    auto cameraUBO = std::make_shared<CameraUboType>();
    InterfaceCameraOrbit orbit(InterfaceCameraOrbit::UpDirection::Y);
    orbit.initialize(*vulkanContext);
    orbit.setAzimuth(0.9f); orbit.setElevation(-0.5f);
    orbit.setPosition({-0.5f, 0.0f, 0.5f}); orbit.setDistance(1.0f);
    orbit.update(cameraUBO->ubo);

    auto splatting = vulkanContext->create<VulkanGaussianSplatting>(
        imageViewSrc, cameraUBO, spzPath);
    engine.add(splatting);

    for (uint32_t i = 0; i < vulkanContext->getNumberOfSwapChainImages(); ++i)
        cameraUBO->update(i);

    const int FRAMES = 5;
    for (int f = 0; f < FRAMES; ++f) {
        engine.step();
        vkQueueWaitIdle(vulkanContext->getGraphicsQueue());
    }

    VkExtent2D ext = vulkanContext->getSwapChainExtent();
    auto pixels = readImageToHost(*vulkanContext, imgs[0], ext.width, ext.height);
    writePPM("test_gaussian_splatting_render.ppm", pixels.data(), ext.width, ext.height);

    uint8_t maxVal = *std::max_element(pixels.begin(), pixels.end());
    std::cout << "\n  classWithRaccoonScene: image max=" << (int)maxVal << "\n";

    std::cout << "--- GPU profiling (mean over " << FRAMES << " frames) ---\n";
    for (auto& [name, ms] : engine.getProfilingResults())
        std::cout << "  " << name << ": " << ms << " ms\n";

    EXPECT_GT(maxVal, uint8_t(10)) << "Rendered image is all-black — pipeline drew nothing";
}

// ----------------------------------------------------------------
// Test: classWithRaccoonTwoFrames
// Uses VulkanGaussianSplatting class with the raccoon SPZ file.
// Renders 4 frames (path 0, 1, 0, 1) and checks:
//   1. Bit-exact determinism: path 0 run 1 == path 0 run 2 (and same for path 1)
//   2. Content: at least 12 of 16 bins in the 4×4 grid have non-zero pixels
// ----------------------------------------------------------------
TEST_F(GaussianSplattingTest, classWithRaccoonTwoFrames) {
    const std::string spzPath = "3rdparty/spz/samples/racoonfamily.spz";
    if (!std::filesystem::exists(spzPath)) {
        GTEST_SKIP() << "SPZ sample not found: " << spzPath;
    }

    auto& engine = frontend->getKlartraumEngine();

    std::vector<VkImage> imgs;
    auto imageViewSrc = makeImageViewSrc(*vulkanContext, imgs);
    for (uint32_t i = 0; i < vulkanContext->getNumberOfSwapChainImages(); ++i)
        imageViewSrc->setWaitFor(i, vulkanContext->imageAvailableSemaphoresPerImage[i]);

    auto cameraUBO = std::make_shared<CameraUboType>();
    InterfaceCameraOrbit orbit(InterfaceCameraOrbit::UpDirection::Y);
    orbit.initialize(*vulkanContext);
    orbit.setAzimuth(0.9f); orbit.setElevation(-0.5f);
    orbit.setPosition({-0.5f, 0.0f, 0.5f}); orbit.setDistance(1.0f);
    orbit.update(cameraUBO->ubo);

    auto splatting = vulkanContext->create<VulkanGaussianSplatting>(
        imageViewSrc, cameraUBO, spzPath);
    engine.add(splatting);

    for (uint32_t i = 0; i < vulkanContext->getNumberOfSwapChainImages(); ++i)
        cameraUBO->update(i);

    VkExtent2D ext = vulkanContext->getSwapChainExtent();

    engine.step(); vkQueueWaitIdle(vulkanContext->getGraphicsQueue());
    auto frame1 = readImageToHost(*vulkanContext, imgs[0], ext.width, ext.height);

    engine.step(); vkQueueWaitIdle(vulkanContext->getGraphicsQueue());
    auto frame2 = readImageToHost(*vulkanContext, imgs[1], ext.width, ext.height);

    engine.step(); vkQueueWaitIdle(vulkanContext->getGraphicsQueue());
    auto frame3 = readImageToHost(*vulkanContext, imgs[0], ext.width, ext.height);

    engine.step(); vkQueueWaitIdle(vulkanContext->getGraphicsQueue());
    auto frame4 = readImageToHost(*vulkanContext, imgs[1], ext.width, ext.height);

    writePPM("test_gsplatting_frame1.ppm", frame1.data(), ext.width, ext.height);
    writePPM("test_gsplatting_frame2.ppm", frame2.data(), ext.width, ext.height);
    writePPM("test_gsplatting_frame3.ppm", frame3.data(), ext.width, ext.height);
    writePPM("test_gsplatting_frame4.ppm", frame4.data(), ext.width, ext.height);

    // Temporal determinism: same path must be bit-exact across frames
    EXPECT_EQ(frame1, frame3)
        << "Path 0: frame 1 and frame 3 differ — rendering is not deterministic";
    EXPECT_EQ(frame2, frame4)
        << "Path 1: frame 2 and frame 4 differ — rendering is not deterministic";

    // Content check: raccoon should fill most of the 4×4 bin grid
    const uint32_t gridSize = 4;
    const uint32_t cellW = ext.width  / gridSize;
    const uint32_t cellH = ext.height / gridSize;
    int binsWithContent = 0;
    uint8_t maxBright = 0;
    std::cout << "  Bin-centre brightness (4×4 grid):\n";
    for (uint32_t by = 0; by < gridSize; ++by) {
        for (uint32_t bx = 0; bx < gridSize; ++bx) {
            uint32_t cx = bx * cellW + cellW / 2;
            uint32_t cy = by * cellH + cellH / 2;
            uint8_t patchMax = 0;
            for (int dy = -2; dy <= 2; ++dy)
                for (int dx = -2; dx <= 2; ++dx) {
                    int px = (int)cx + dx, py = (int)cy + dy;
                    if (px < 0 || px >= (int)ext.width || py < 0 || py >= (int)ext.height) continue;
                    size_t base = ((size_t)py * ext.width + px) * 4;
                    patchMax = std::max(patchMax,
                        std::max({frame1[base], frame1[base+1], frame1[base+2]}));
                }
            if (patchMax > 0) ++binsWithContent;
            maxBright = std::max(maxBright, patchMax);
            std::cout << "    bin(" << bx << "," << by << ") centre("
                      << cx << "," << cy << ") max=" << (int)patchMax << "\n";
        }
    }
    std::cout << "  Bins with content: " << binsWithContent << "/16, max=" << (int)maxBright << "\n";

    EXPECT_GE(binsWithContent, 12)
        << "Only " << binsWithContent << "/16 bins non-zero — possible binning regression";
    EXPECT_GT(maxBright, uint8_t(100)) << "Rendered image too dark";
}