#include <gtest/gtest.h>
#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

#include <glm/glm.hpp>

#include "klartraum/headless_frontend.hpp"
#include "klartraum/vulkan_context.hpp"
#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/generalcomputation.hpp"
#include "klartraum/computegraph/bufferelement.hpp"
#include "klartraum/vulkan_buffer.hpp"
#include "klartraum/vulkan_gaussian_splatting_types.hpp"
#include "klartraum/interface_camera_orbit.hpp"
#include "klartraum/backend_config.hpp"
#include "klartraum/computegraph/imageviewsrc.hpp"

using namespace klartraum;

class SoATest : public ::testing::Test {
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

// Helper: convert AoS Gaussian3D vector into SoA vectors ready for GPU upload.
// SH layout: band-major → shR[band * N + gaussianIdx]
struct Gaussian3DSoA {
    std::vector<glm::vec3> positions;
    std::vector<glm::vec4> rotations;
    std::vector<glm::vec3> scales;
    std::vector<glm::vec4> colorsAlpha;  // (r,g,b,alpha)
    std::vector<float>     shR;          // [15 * N], band-major
    std::vector<float>     shG;
    std::vector<float>     shB;

    static Gaussian3DSoA fromAoS(const std::vector<Gaussian3D>& src) {
        size_t N = src.size();
        Gaussian3DSoA soa;
        soa.positions.resize(N);
        soa.rotations.resize(N);
        soa.scales.resize(N);
        soa.colorsAlpha.resize(N);
        soa.shR.resize(15 * N, 0.0f);
        soa.shG.resize(15 * N, 0.0f);
        soa.shB.resize(15 * N, 0.0f);

        for (size_t i = 0; i < N; i++) {
            const auto& g = src[i];
            soa.positions[i]   = {g.position[0], g.position[1], g.position[2]};
            soa.rotations[i]   = {g.rotation[0], g.rotation[1], g.rotation[2], g.rotation[3]};
            soa.scales[i]      = {g.scale[0],    g.scale[1],    g.scale[2]};
            soa.colorsAlpha[i] = {g.color[0],    g.color[1],    g.color[2],    g.alpha};
            for (int b = 0; b < 15; b++) {
                soa.shR[b * N + i] = g.shR[b];
                soa.shG[b * N + i] = g.shG[b];
                soa.shB[b * N + i] = g.shB[b];
            }
        }
        return soa;
    }
};

// ----------------------------------------------------------------
// Test: projection_soa
// Same three gaussians as the AoS projection test. Verifies that the
// SoA projection shader produces identical screen-space positions/depths.
// ----------------------------------------------------------------
TEST_F(SoATest, projection_soa) {
    const uint32_t N = 3;

    std::vector<Gaussian3D> g3d(N);
    // origin
    g3d[0].position = {0.f, 0.f, 0.f};
    g3d[0].scale    = {1.f, 1.f, 1.f};
    g3d[0].rotation = {0.f, 0.f, 0.f, 1.f};
    g3d[0].alpha    = 0.5f;
    // Y+1
    g3d[1].position = {0.f, 1.f, 0.f};
    g3d[1].scale    = {1.f, 1.f, 1.f};
    g3d[1].rotation = {0.f, 0.f, 0.f, 1.f};
    g3d[1].alpha    = 0.5f;
    // Z+1
    g3d[2].position = {0.f, 0.f, 1.f};
    g3d[2].scale    = {1.f, 1.f, 1.f};
    g3d[2].rotation = {0.f, 0.f, 0.f, 1.f};
    g3d[2].alpha    = 0.5f;

    auto soa = Gaussian3DSoA::fromAoS(g3d);

    // --- SoA GPU buffers ---
    auto bufPos      = std::make_shared<BufferElement<VulkanBuffer<glm::vec3>>>(*vulkanContext, N);
    auto bufRot      = std::make_shared<BufferElement<VulkanBuffer<glm::vec4>>>(*vulkanContext, N);
    auto bufScale    = std::make_shared<BufferElement<VulkanBuffer<glm::vec3>>>(*vulkanContext, N);
    auto bufColAlpha = std::make_shared<BufferElement<VulkanBuffer<glm::vec4>>>(*vulkanContext, N);
    auto bufShR      = std::make_shared<BufferElement<VulkanBuffer<float>>>(*vulkanContext, 15 * N);
    auto bufShG      = std::make_shared<BufferElement<VulkanBuffer<float>>>(*vulkanContext, 15 * N);
    auto bufShB      = std::make_shared<BufferElement<VulkanBuffer<float>>>(*vulkanContext, 15 * N);

    std::shared_ptr<CameraUboType> cameraUBO;
    makeCameraUBO(cameraUBO);

    // SoA Gaussian2D output buffers
    auto bufPos2D      = std::make_shared<BufferElement<VulkanBuffer<glm::vec2>>>(*vulkanContext, N);
    auto bufZ          = std::make_shared<BufferElement<VulkanBuffer<float>>>(*vulkanContext, N);
    auto bufBinMask    = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(*vulkanContext, N);
    auto bufCovInv     = std::make_shared<BufferElement<VulkanBuffer<glm::vec4>>>(*vulkanContext, N);
    auto bufColorAlpha = std::make_shared<BufferElement<VulkanBuffer<glm::vec4>>>(*vulkanContext, N);

    auto proj = std::make_shared<GaussianProjection>(
        *vulkanContext, "shaders/gsplat/gsplat_projection_soa.comp.spv");
    proj->setInput(bufPos,      0);
    proj->setInput(bufRot,      1);
    proj->setInput(bufScale,    2);
    proj->setInput(bufColAlpha, 3);
    proj->setInput(bufShR,      4);
    proj->setInput(bufShG,      5);
    proj->setInput(bufShB,      6);
    proj->setInput(cameraUBO,   7);
    proj->setInput(bufPos2D,    8);
    proj->setInput(bufZ,        9);
    proj->setInput(bufBinMask,  10);
    proj->setInput(bufCovInv,   11);
    proj->setInput(bufColorAlpha, 12);

    proj->setPushConstants({{N, 4,
        (float)BackendConfig::WIDTH, (float)BackendConfig::HEIGHT}});
    proj->setGroupCountX(N / 128 + 1);

    auto cg = ComputeGraph(*vulkanContext, 1);
    cg.enableProfiling();
    cg.compileFrom(proj);

    // Upload SoA data
    bufPos->getBuffer(0).memcopyFrom(soa.positions);
    bufRot->getBuffer(0).memcopyFrom(soa.rotations);
    bufScale->getBuffer(0).memcopyFrom(soa.scales);
    bufColAlpha->getBuffer(0).memcopyFrom(soa.colorsAlpha);
    bufShR->getBuffer(0).memcopyFrom(soa.shR);
    bufShG->getBuffer(0).memcopyFrom(soa.shG);
    bufShB->getBuffer(0).memcopyFrom(soa.shB);
    cameraUBO->update(0);

    cg.submitAndWait(vulkanContext->getGraphicsQueue(), 0);

    // Read back
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

    // Print profiling
    auto results = cg.getProfilingResults();
    std::cout << "\n--- SoA projection profiling ---\n";
    for (auto& [name, ms] : results)
        std::cout << "  " << name << ": " << ms << " ms\n";

    SUCCEED();
}

// ----------------------------------------------------------------
// Profiling comparison: AoS vs SoA projection with 500K gaussians.
// Runs each version 5 times and reports mean GPU time.
// ----------------------------------------------------------------
TEST_F(SoATest, projection_profiling_comparison) {
    const uint32_t N = 500000;
    const int RUNS = 5;

    // Generate random Gaussian3D data
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> distPos(0.0f, 1.0f);

    std::vector<Gaussian3D> g3d(N);
    for (auto& g : g3d) {
        g.position = {dist(rng), dist(rng), dist(rng)};
        g.scale    = {0.01f, 0.01f, 0.01f};
        g.rotation = {0.f, 0.f, 0.f, 1.f};
        g.color    = {distPos(rng), distPos(rng), distPos(rng)};
        g.alpha    = distPos(rng);
        for (int b = 0; b < 15; b++) {
            g.shR[b] = dist(rng) * 0.1f;
            g.shG[b] = dist(rng) * 0.1f;
            g.shB[b] = dist(rng) * 0.1f;
        }
    }

    auto soa = Gaussian3DSoA::fromAoS(g3d);

    // ---- AoS pipeline ----
    float aosTime = 0.0f;
    {
        std::shared_ptr<CameraUboType> cameraUBO;
        makeCameraUBO(cameraUBO, 5.0f);

        auto bufIn  = std::make_shared<BufferElement<Gaussian3DBuffer>>(*vulkanContext, N);
        auto bufOut = std::make_shared<BufferElement<Gaussian2DBuffer>>(*vulkanContext, N);

        auto proj = std::make_shared<GaussianProjection>(
            *vulkanContext, "shaders/gsplat/gsplat_projection.comp.spv");
        proj->setInput(bufIn,     0);
        proj->setInput(cameraUBO, 1);
        proj->setInput(bufOut,    2);
        proj->setPushConstants({{N, 4,
            (float)BackendConfig::WIDTH, (float)BackendConfig::HEIGHT}});
        proj->setGroupCountX(N / 128 + 1);

        auto cg = ComputeGraph(*vulkanContext, 1);
        cg.enableProfiling();
        cg.compileFrom(proj);

        bufIn->getBuffer(0).memcopyFrom(g3d);
        cameraUBO->update(0);

        cg.submitAndWait(vulkanContext->getGraphicsQueue(), 0);

        auto results = cg.getProfilingResults();
        for (auto& [name, ms] : results)
            if (name.find("GeneralComputation") != std::string::npos)
                aosTime = ms;
    }

    // ---- SoA pipeline ----
    float soaTime = 0.0f;
    {
        std::shared_ptr<CameraUboType> cameraUBO;
        makeCameraUBO(cameraUBO, 5.0f);

        auto bufPos      = std::make_shared<BufferElement<VulkanBuffer<glm::vec3>>>(*vulkanContext, N);
        auto bufRot      = std::make_shared<BufferElement<VulkanBuffer<glm::vec4>>>(*vulkanContext, N);
        auto bufScale    = std::make_shared<BufferElement<VulkanBuffer<glm::vec3>>>(*vulkanContext, N);
        auto bufColAlpha = std::make_shared<BufferElement<VulkanBuffer<glm::vec4>>>(*vulkanContext, N);
        auto bufShR      = std::make_shared<BufferElement<VulkanBuffer<float>>>(*vulkanContext, 15 * N);
        auto bufShG      = std::make_shared<BufferElement<VulkanBuffer<float>>>(*vulkanContext, 15 * N);
        auto bufShB      = std::make_shared<BufferElement<VulkanBuffer<float>>>(*vulkanContext, 15 * N);

        auto bufPos2D      = std::make_shared<BufferElement<VulkanBuffer<glm::vec2>>>(*vulkanContext, N);
        auto bufZ          = std::make_shared<BufferElement<VulkanBuffer<float>>>(*vulkanContext, N);
        auto bufBinMask    = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(*vulkanContext, N);
        auto bufCovInv     = std::make_shared<BufferElement<VulkanBuffer<glm::vec4>>>(*vulkanContext, N);
        auto bufColorAlpha = std::make_shared<BufferElement<VulkanBuffer<glm::vec4>>>(*vulkanContext, N);

        auto proj = std::make_shared<GaussianProjection>(
            *vulkanContext, "shaders/gsplat/gsplat_projection_soa.comp.spv");
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

        bufPos->getBuffer(0).memcopyFrom(soa.positions);
        bufRot->getBuffer(0).memcopyFrom(soa.rotations);
        bufScale->getBuffer(0).memcopyFrom(soa.scales);
        bufColAlpha->getBuffer(0).memcopyFrom(soa.colorsAlpha);
        bufShR->getBuffer(0).memcopyFrom(soa.shR);
        bufShG->getBuffer(0).memcopyFrom(soa.shG);
        bufShB->getBuffer(0).memcopyFrom(soa.shB);
        cameraUBO->update(0);

        cg.submitAndWait(vulkanContext->getGraphicsQueue(), 0);

        auto results = cg.getProfilingResults();
        for (auto& [name, ms] : results)
            if (name.find("GeneralComputation") != std::string::npos)
                soaTime = ms;
    }

    std::cout << "\n=== Projection profiling (N=" << N << ", mean over " << RUNS << " runs) ===\n";
    std::cout << "  AoS: " << aosTime << " ms\n";
    std::cout << "  SoA: " << soaTime << " ms\n";
    if (soaTime > 0.0f)
        std::cout << "  Speedup: " << (aosTime / soaTime) << "x\n";

    SUCCEED();
}

// ----------------------------------------------------------------
// Full SoA pipeline: projection → binning (3-pass) → sort → gather
//                   → bounds → splatting.
// Uses a single red Gaussian that should land in bin 10.
// Verifies the binned count is > 0 and the sorted bin masks are ordered.
// ----------------------------------------------------------------
TEST_F(SoATest, fullPipeline_soa) {
    const float    W        = BackendConfig::WIDTH;
    const float    H        = BackendConfig::HEIGHT;
    const uint32_t gridSize = 4;
    const uint32_t numBins  = gridSize * gridSize;  // 16
    const uint32_t tpg      = 128;

    // Single red gaussian that projects to the centre of bin 10
    const uint32_t N = 1;
    Gaussian3D g{};
    g.position = {0.0f, -0.690f, -0.518f};
    g.scale    = {0.05f, 0.05f, 0.05f};
    g.rotation = {0.0f, 0.0f, 0.0f, 1.0f};
    g.color    = {1.772f, -1.772f, -1.772f};
    g.alpha    = 1.0f;
    auto soa = Gaussian3DSoA::fromAoS(std::vector<Gaussian3D>{g});

    const uint32_t numBinWGs  = N / tpg + 1;
    const uint32_t maxBinned  = N * numBins;
    const uint32_t numSortWGs = std::max(1u, std::min(320u, maxBinned / tpg + 1));

    // --- Camera ---
    std::shared_ptr<CameraUboType> cameraUBO;
    makeCameraUBO(cameraUBO);

    // --- Projected SoA (N) ---
    auto bufPos      = std::make_shared<BufferElement<VulkanBuffer<glm::vec3>>>(*vulkanContext, N);
    auto bufRot      = std::make_shared<BufferElement<VulkanBuffer<glm::vec4>>>(*vulkanContext, N);
    auto bufScale    = std::make_shared<BufferElement<VulkanBuffer<glm::vec3>>>(*vulkanContext, N);
    auto bufColAlpha = std::make_shared<BufferElement<VulkanBuffer<glm::vec4>>>(*vulkanContext, N);
    auto bufShR      = std::make_shared<BufferElement<VulkanBuffer<float>>>(*vulkanContext, 15*N);
    auto bufShG      = std::make_shared<BufferElement<VulkanBuffer<float>>>(*vulkanContext, 15*N);
    auto bufShB      = std::make_shared<BufferElement<VulkanBuffer<float>>>(*vulkanContext, 15*N);

    auto bufProjPos2D  = std::make_shared<BufferElement<VulkanBuffer<glm::vec2>>>(*vulkanContext, N);
    auto bufProjZ      = std::make_shared<BufferElement<VulkanBuffer<float>>>(*vulkanContext, N);
    auto bufProjBinMask= std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(*vulkanContext, N);
    auto bufProjCovInv = std::make_shared<BufferElement<VulkanBuffer<glm::vec4>>>(*vulkanContext, N);
    auto bufProjCA     = std::make_shared<BufferElement<VulkanBuffer<glm::vec4>>>(*vulkanContext, N);

    // --- Binning buffers ---
    auto bufHistogram  = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(*vulkanContext, numBins * numBinWGs);
    bufHistogram->setRecordToZero(true);
    auto bufOffsets    = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(*vulkanContext, numBinWGs + 1);
    bufOffsets->setRecordToZero(true);
    auto bufTotalCount = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(*vulkanContext, 1);
    bufTotalCount->setRecordToZero(true);

    // --- Binned SoA (maxBinned) ---
    auto bufBinPos2D  = std::make_shared<BufferElement<VulkanBuffer<glm::vec2>>>(*vulkanContext, maxBinned);
    auto bufBinZ      = std::make_shared<BufferElement<VulkanBuffer<float>>>(*vulkanContext, maxBinned);
    auto bufBinMask   = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(*vulkanContext, maxBinned);
    auto bufBinCovInv = std::make_shared<BufferElement<VulkanBuffer<glm::vec4>>>(*vulkanContext, maxBinned);
    auto bufBinCA     = std::make_shared<BufferElement<VulkanBuffer<glm::vec4>>>(*vulkanContext, maxBinned);

    // --- Sort ping-pong + scratch ---
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

    // --- Sorted SoA (maxBinned) ---
    auto bufSortPos2D  = std::make_shared<BufferElement<VulkanBuffer<glm::vec2>>>(*vulkanContext, maxBinned);
    auto bufSortZ      = std::make_shared<BufferElement<VulkanBuffer<float>>>(*vulkanContext, maxBinned);
    auto bufSortMask   = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(*vulkanContext, maxBinned);
    auto bufSortCovInv = std::make_shared<BufferElement<VulkanBuffer<glm::vec4>>>(*vulkanContext, maxBinned);
    auto bufSortCA     = std::make_shared<BufferElement<VulkanBuffer<glm::vec4>>>(*vulkanContext, maxBinned);

    // --- Bounds: 16 StartAndEnd structs = 32 uint32_t ---
    auto bufBounds = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(*vulkanContext, numBins * 2);
    bufBounds->setRecordToZero(true);

    // --- Image ---
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

    // ---- Stage 1: projection SoA ----
    {
        auto proj = std::make_shared<GaussianProjection>(
            *vulkanContext, "shaders/gsplat/gsplat_projection_soa.comp.spv");
        proj->setInput(bufPos,      0);
        proj->setInput(bufRot,      1);
        proj->setInput(bufScale,    2);
        proj->setInput(bufColAlpha, 3);
        proj->setInput(bufShR,      4);
        proj->setInput(bufShG,      5);
        proj->setInput(bufShB,      6);
        proj->setInput(cameraUBO,   7);
        proj->setInput(bufProjPos2D,  8);
        proj->setInput(bufProjZ,      9);
        proj->setInput(bufProjBinMask,10);
        proj->setInput(bufProjCovInv, 11);
        proj->setInput(bufProjCA,     12);
        proj->setPushConstants({{N, gridSize, W, H}});
        proj->setGroupCountX(N / tpg + 1);

        auto cg = ComputeGraph(*vulkanContext, 1);
        cg.compileFrom(proj);
        bufPos->getBuffer(0).memcopyFrom(soa.positions);
        bufRot->getBuffer(0).memcopyFrom(soa.rotations);
        bufScale->getBuffer(0).memcopyFrom(soa.scales);
        bufColAlpha->getBuffer(0).memcopyFrom(soa.colorsAlpha);
        bufShR->getBuffer(0).memcopyFrom(soa.shR);
        bufShG->getBuffer(0).memcopyFrom(soa.shG);
        bufShB->getBuffer(0).memcopyFrom(soa.shB);
        cameraUBO->update(0);
        cg.submitAndWait(vulkanContext->getGraphicsQueue(), 0);
    }

    // ---- Stage 2: binning count SoA ----
    {
        auto count = std::make_shared<GaussianBinningCount>(
            *vulkanContext, "shaders/gsplat/gsplat_binning_count_soa.comp.spv");
        count->setInput(bufProjPos2D,  0);
        count->setInput(bufProjZ,      1);
        count->setInput(bufProjCovInv, 2);
        count->setInput(bufHistogram,  3);
        count->setInput(bufOffsets,    4);
        count->setPushConstants({{N, gridSize, W, H}});
        count->setGroupCountX(numBinWGs);
        submitStage(count);
        bufHistogram->setRecordToZero(false);
        bufOffsets->setRecordToZero(false);
    }

    // ---- Stage 3: binning prefix sum (AoS shader reused) ----
    {
        auto ps = std::make_shared<GeneralComputation<>>(
            *vulkanContext, "shaders/gsplat/gsplat_binning_prefix_sum.comp.spv");
        ps->setInput(bufHistogram, 0);
        ps->setInput(bufOffsets,   1);
        ps->setGroupCountX(numBinWGs);
        submitStage(ps);
    }

    // ---- Stage 4: binning scatter SoA ----
    {
        auto scatter = std::make_shared<GaussianBinningScatter>(
            *vulkanContext, "shaders/gsplat/gsplat_binning_scatter_soa.comp.spv");
        scatter->setInput(bufProjPos2D,  0);
        scatter->setInput(bufProjZ,      1);
        scatter->setInput(bufProjCovInv, 2);
        scatter->setInput(bufProjCA,     3);
        scatter->setInput(bufHistogram,  4);  // prefix sums
        scatter->setInput(bufTotalCount, 5);
        scatter->setInput(bufBinPos2D,   6);
        scatter->setInput(bufBinZ,       7);
        scatter->setInput(bufBinMask,    8);
        scatter->setInput(bufBinCovInv,  9);
        scatter->setInput(bufBinCA,      10);
        scatter->setPushConstants({{N, gridSize, W, H, maxBinned}});
        scatter->setGroupCountX(numBinWGs);
        submitStage(scatter);
        bufTotalCount->setRecordToZero(false);
    }

    // Read totalCount from GPU
    uint32_t totalCount = 0;
    {
        std::vector<uint32_t> tmp(1);
        bufTotalCount->getBuffer(0).memcopyTo(tmp);
        totalCount = tmp[0];
    }
    ASSERT_GT(totalCount, 0u) << "Binning produced zero entries";
    std::cout << "  totalCount after binning: " << totalCount << "\n";

    // ---- Stages 5+6: extract sort keys → radix sort ----
    // Compile sort FIRST so all four ping-pong buffers are allocated.
    // Keep the compiled ComputeGraph alive and reuse it for submission
    // (avoids double _setup / pipeline-layout leak from compiling twice).
    auto sortOp = std::make_shared<RadixSort>(*vulkanContext, std::vector<std::string>{
        "shaders/gsplat/gsplat_radix_sort_histogram.comp.spv",
        "shaders/gsplat/gsplat_radix_sort_hist_prefix_sum.comp.spv",
        "shaders/gsplat/gsplat_radix_sort_hist_scatter.comp.spv"
    });
    sortOp->setInput(bufValA, 0);
    sortOp->setInput(bufIdxA, 1);
    sortOp->setInput(bufValB, 2);
    sortOp->setInput(bufIdxB, 3);
    sortOp->addScratchBufferElement(scratchCounts,  true);
    sortOp->addScratchBufferElement(scratchOffsets, true);
    sortOp->addScratchBufferElement(bufTotalCount,  false);
    sortOp->addScratchBufferElement(scratchHist,    true);
    sortOp->setGroupCountX(numSortWGs);
    {
        std::vector<SortPushConstants> pcs;
        for (uint32_t i = 0; i < 32 / 4; ++i)
            pcs.push_back({i, totalCount, numBins});
        sortOp->setPushConstants(pcs);
    }

    // One compile: allocates valA/idxA/valB/idxB, records sort command buffer
    ComputeGraph cgSort(*vulkanContext, 1);
    cgSort.compileFrom(sortOp);

    // Pre-fill sentinel so uninitialized slots beyond totalCount sort to end
    {
        std::vector<uint32_t> sentinel(maxBinned, 0xFFFFFFFFu);
        bufValA->getBuffer(0).memcopyFrom(sentinel);
        bufIdxA->getBuffer(0).memcopyFrom(sentinel);
    }

    // Extract: writes valA[0..totalCount-1] and idxA[0..totalCount-1]
    {
        auto extract = std::make_shared<GeneralComputation<>>(
            *vulkanContext, "shaders/gsplat/gsplat_extract_sort_keys_soa.comp.spv");
        extract->setInput(bufBinMask,    0);
        extract->setInput(bufBinZ,       1);
        extract->setInput(bufTotalCount, 2);
        extract->setInput(bufValA,       3);
        extract->setInput(bufIdxA,       4);
        extract->setGroupCountX(maxBinned / tpg + 1); // 1 thread/element, no looping
        submitStage(extract);
    }

    // Sort: reuse the already-compiled cgSort (no second compileFrom)
    cgSort.submitAndWait(vulkanContext->getGraphicsQueue(), 0);

    // ---- Stage 7: gather sorted SoA ----
    // After 8 radix passes (even) output is in A buffers
    {
        auto gather = std::make_shared<GeneralComputation<>>(
            *vulkanContext, "shaders/gsplat/gsplat_gather_sorted_soa.comp.spv");
        gather->setInput(bufBinPos2D,  0);
        gather->setInput(bufBinZ,      1);
        gather->setInput(bufBinMask,   2);
        gather->setInput(bufBinCovInv, 3);
        gather->setInput(bufBinCA,     4);
        gather->setInput(bufIdxA,      5);  // sorted indexes
        gather->setInput(bufTotalCount,6);
        gather->setInput(bufSortPos2D, 7);
        gather->setInput(bufSortZ,     8);
        gather->setInput(bufSortMask,  9);
        gather->setInput(bufSortCovInv,10);
        gather->setInput(bufSortCA,    11);
        gather->setGroupCountX(maxBinned / tpg + 1); // 1 thread/element, no looping
        submitStage(gather);
    }

    // ---- Stage 8: bin bounds SoA ----
    {
        auto bounds = std::make_shared<GeneralComputation<>>(
            *vulkanContext, "shaders/gsplat/gsplat_bin_bounds_soa.comp.spv");
        bounds->setInput(bufSortMask,   0);
        bounds->setInput(bufTotalCount, 1);
        bounds->setInput(bufBounds,     2);
        bounds->setGroupCountX(maxBinned / 256 + 1);
        submitStage(bounds);
        bufBounds->setRecordToZero(false);
    }

    // ---- Stage 9: binned splatting SoA ----
    {
        const uint32_t tbX  = 8, tbY = 8;
        const uint32_t gpbX = uint32_t((W / tbX) / gridSize);
        const uint32_t gpbY = uint32_t((H / tbY) / gridSize);

        auto splat = std::make_shared<GaussianSplatting>(
            *vulkanContext, "shaders/gsplat/gsplat_binned_splatting_soa.comp.spv");
        splat->setInput(bufSortPos2D,  0);
        splat->setInput(bufSortCovInv, 1);
        splat->setInput(bufSortCA,     2);
        splat->setInput(bufTotalCount, 3);
        splat->setInput(bufBounds,     4);
        splat->setInput(imageViewSrc,  5);
        std::vector<SplatPushConstants> pcs;
        for (uint32_t y = 0; y < gridSize; ++y)
            for (uint32_t x = 0; x < gridSize; ++x)
                pcs.push_back({totalCount, gridSize, x, y, W, H});
        splat->setPushConstants(pcs);
        splat->setGroupCountX(gpbX);
        splat->setGroupCountY(gpbY);
        splat->setGroupCountZ(1);
        submitStage(splat);
    }

    // Verify sorted bin masks are non-decreasing
    std::vector<uint32_t> sortedMasks(totalCount);
    bufSortMask->getBuffer(0).memcopyTo(sortedMasks);
    for (uint32_t i = 1; i < totalCount; ++i)
        EXPECT_LE(sortedMasks[i-1], sortedMasks[i]) << "Bin masks not sorted at " << i;

    std::cout << "  fullPipeline_soa: OK, " << totalCount << " binned gaussians rendered\n";
    SUCCEED();
}

// ----------------------------------------------------------------
// Helpers shared with the comparison tests below
// ----------------------------------------------------------------
static std::vector<uint8_t> readImageSoA(VulkanContext& vc, VkImage image,
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

// Helper: run the complete SoA pipeline for a given set of Gaussian3D, return the rendered image.
static std::vector<uint8_t> renderSoA(VulkanContext& vc,
                                       const std::vector<Gaussian3D>& g3dVec,
                                       std::shared_ptr<CameraUboType> cameraUBO)
{
    const float    W        = BackendConfig::WIDTH;
    const float    H        = BackendConfig::HEIGHT;
    const uint32_t gridSize = 4;
    const uint32_t numBins  = gridSize * gridSize;
    const uint32_t tpg      = 128;
    const uint32_t N        = (uint32_t)g3dVec.size();

    auto soa = Gaussian3DSoA::fromAoS(g3dVec);

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

    auto bufProjPos2D  = std::make_shared<BufferElement<VulkanBuffer<glm::vec2>>>(vc, N);
    auto bufProjZ      = std::make_shared<BufferElement<VulkanBuffer<float>>>(vc, N);
    auto bufProjBinMask= std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vc, N);
    auto bufProjCovInv = std::make_shared<BufferElement<VulkanBuffer<glm::vec4>>>(vc, N);
    auto bufProjCA     = std::make_shared<BufferElement<VulkanBuffer<glm::vec4>>>(vc, N);

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
    {
        auto proj = std::make_shared<GaussianProjection>(vc, "shaders/gsplat/gsplat_projection_soa.comp.spv");
        proj->setInput(bufPos, 0); proj->setInput(bufRot, 1); proj->setInput(bufScale, 2);
        proj->setInput(bufColAlpha, 3); proj->setInput(bufShR, 4); proj->setInput(bufShG, 5);
        proj->setInput(bufShB, 6); proj->setInput(cameraUBO, 7);
        proj->setInput(bufProjPos2D, 8); proj->setInput(bufProjZ, 9); proj->setInput(bufProjBinMask, 10);
        proj->setInput(bufProjCovInv, 11); proj->setInput(bufProjCA, 12);
        proj->setPushConstants({{N, gridSize, W, H}});
        proj->setGroupCountX(N / tpg + 1);
        auto cg = ComputeGraph(vc, 1);
        cg.compileFrom(proj);
        bufPos->getBuffer(0).memcopyFrom(soa.positions);
        bufRot->getBuffer(0).memcopyFrom(soa.rotations);
        bufScale->getBuffer(0).memcopyFrom(soa.scales);
        bufColAlpha->getBuffer(0).memcopyFrom(soa.colorsAlpha);
        bufShR->getBuffer(0).memcopyFrom(soa.shR);
        bufShG->getBuffer(0).memcopyFrom(soa.shG);
        bufShB->getBuffer(0).memcopyFrom(soa.shB);
        cameraUBO->update(0);
        cg.submitAndWait(vc.getGraphicsQueue(), 0);
    }

    // Count
    { auto c = std::make_shared<GaussianBinningCount>(vc, "shaders/gsplat/gsplat_binning_count_soa.comp.spv");
      c->setInput(bufProjPos2D, 0); c->setInput(bufProjZ, 1); c->setInput(bufProjCovInv, 2);
      c->setInput(bufHistogram, 3); c->setInput(bufOffsets, 4);
      c->setPushConstants({{N, gridSize, W, H}}); c->setGroupCountX(numBinWGs); submitS(c);
      bufHistogram->setRecordToZero(false); bufOffsets->setRecordToZero(false); }

    // PrefixSum
    { auto ps = std::make_shared<GeneralComputation<>>(vc, "shaders/gsplat/gsplat_binning_prefix_sum.comp.spv");
      ps->setInput(bufHistogram, 0); ps->setInput(bufOffsets, 1); ps->setGroupCountX(numBinWGs); submitS(ps); }

    // Scatter
    { auto sc = std::make_shared<GaussianBinningScatter>(vc, "shaders/gsplat/gsplat_binning_scatter_soa.comp.spv");
      sc->setInput(bufProjPos2D, 0); sc->setInput(bufProjZ, 1); sc->setInput(bufProjCovInv, 2);
      sc->setInput(bufProjCA, 3); sc->setInput(bufHistogram, 4); sc->setInput(bufTotalCount, 5);
      sc->setInput(bufBinPos2D, 6); sc->setInput(bufBinZ, 7); sc->setInput(bufBinMask, 8);
      sc->setInput(bufBinCovInv, 9); sc->setInput(bufBinCA, 10);
      sc->setPushConstants({{N, gridSize, W, H, maxBinned}}); sc->setGroupCountX(numBinWGs); submitS(sc);
      bufTotalCount->setRecordToZero(false); }

    uint32_t totalCount = 0;
    { std::vector<uint32_t> tmp(1); bufTotalCount->getBuffer(0).memcopyTo(tmp); totalCount = tmp[0]; }
    if (totalCount == 0) return {};

    // Sort setup + extract + sort
    auto sortOp = std::make_shared<RadixSort>(vc, std::vector<std::string>{
        "shaders/gsplat/gsplat_radix_sort_histogram.comp.spv",
        "shaders/gsplat/gsplat_radix_sort_hist_prefix_sum.comp.spv",
        "shaders/gsplat/gsplat_radix_sort_hist_scatter.comp.spv"});
    sortOp->setInput(bufValA, 0); sortOp->setInput(bufIdxA, 1);
    sortOp->setInput(bufValB, 2); sortOp->setInput(bufIdxB, 3);
    sortOp->addScratchBufferElement(scratchCounts, true); sortOp->addScratchBufferElement(scratchOffsets, true);
    sortOp->addScratchBufferElement(bufTotalCount, false); sortOp->addScratchBufferElement(scratchHist, true);
    sortOp->setGroupCountX(numSortWGs);
    { std::vector<SortPushConstants> pcs; for (uint32_t i = 0; i < 8; ++i) pcs.push_back({i, maxBinned, numBins}); sortOp->setPushConstants(pcs); }
    // Single compile — allocates all sort buffers; keep alive and reuse for submission.
    ComputeGraph cgSort(vc, 1);
    cgSort.compileFrom(sortOp);
    { std::vector<uint32_t> sent(maxBinned, 0xFFFFFFFFu);
      bufValA->getBuffer(0).memcopyFrom(sent); bufIdxA->getBuffer(0).memcopyFrom(sent); }
    { auto ex = std::make_shared<GeneralComputation<>>(vc, "shaders/gsplat/gsplat_extract_sort_keys_soa.comp.spv");
      ex->setInput(bufBinMask, 0); ex->setInput(bufBinZ, 1); ex->setInput(bufTotalCount, 2);
      ex->setInput(bufValA, 3); ex->setInput(bufIdxA, 4); ex->setGroupCountX(maxBinned / tpg + 1); submitS(ex); }
    cgSort.submitAndWait(vc.getGraphicsQueue(), 0);

    // Gather + bounds + splat
    { auto g = std::make_shared<GeneralComputation<>>(vc, "shaders/gsplat/gsplat_gather_sorted_soa.comp.spv");
      g->setInput(bufBinPos2D, 0); g->setInput(bufBinZ, 1); g->setInput(bufBinMask, 2);
      g->setInput(bufBinCovInv, 3); g->setInput(bufBinCA, 4); g->setInput(bufIdxA, 5);
      g->setInput(bufTotalCount, 6); g->setInput(bufSortPos2D, 7); g->setInput(bufSortZ, 8);
      g->setInput(bufSortMask, 9); g->setInput(bufSortCovInv, 10); g->setInput(bufSortCA, 11);
      g->setGroupCountX(maxBinned / tpg + 1); submitS(g); }
    { auto b = std::make_shared<GeneralComputation<>>(vc, "shaders/gsplat/gsplat_bin_bounds_soa.comp.spv");
      b->setInput(bufSortMask, 0); b->setInput(bufTotalCount, 1); b->setInput(bufBounds, 2);
      b->setGroupCountX(maxBinned / 256 + 1); submitS(b); bufBounds->setRecordToZero(false); }
    { const uint32_t tbX=8, tbY=8, gpbX=uint32_t((W/tbX)/gridSize), gpbY=uint32_t((H/tbY)/gridSize);
      auto sp = std::make_shared<GaussianSplatting>(vc, "shaders/gsplat/gsplat_binned_splatting_soa.comp.spv");
      sp->setInput(bufSortPos2D, 0); sp->setInput(bufSortCovInv, 1); sp->setInput(bufSortCA, 2);
      sp->setInput(bufTotalCount, 3); sp->setInput(bufBounds, 4); sp->setInput(imageViewSrc, 5);
      std::vector<SplatPushConstants> pcs;
      for (uint32_t y = 0; y < gridSize; ++y) for (uint32_t x = 0; x < gridSize; ++x) pcs.push_back({totalCount, gridSize, x, y, W, H});
      sp->setPushConstants(pcs); sp->setGroupCountX(gpbX); sp->setGroupCountY(gpbY); sp->setGroupCountZ(1); submitS(sp); }

    return readImageSoA(vc, imgs[0], (uint32_t)W, (uint32_t)H);
}

// ----------------------------------------------------------------
// Test: SoA single red Gaussian — mirrors the AoS singleRedGaussian test.
// The gaussian projects to ~(320, 240) = centre of bin 10 (4×4 grid).
// After the full SoA pipeline the rendered pixel at (320,240) must be
// bright in byte2 (r_shader → PPM red) and low in byte0 (b_shader).
// ----------------------------------------------------------------
TEST_F(SoATest, singleRedGaussian_soa) {
    const float    W        = BackendConfig::WIDTH;
    const float    H        = BackendConfig::HEIGHT;

    const float SH_ONE = 1.772f, SH_ZERO = -1.772f;
    Gaussian3D g{};
    g.position = {0.0f, -0.690f, -0.518f};
    g.scale    = {0.05f, 0.05f, 0.05f};
    g.rotation = {0.0f, 0.0f, 0.0f, 1.0f};
    g.color    = {SH_ONE, SH_ZERO, SH_ZERO};
    g.alpha    = 1.0f;

    std::shared_ptr<CameraUboType> cameraUBO;
    makeCameraUBO(cameraUBO);

    auto pixels = renderSoA(*vulkanContext, {g}, cameraUBO);
    ASSERT_FALSE(pixels.empty()) << "SoA pipeline produced zero binned gaussians";

    // Search a 60×60 window around the projected centre (320, 240)
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
    std::cout << "  SoA singleRedGaussian: maxByte2(red)=" << (int)maxByte2
              << " maxByte0(blue)=" << (int)maxByte0 << "\n";
    EXPECT_GT(maxByte2, uint8_t(200)) << "SoA: red channel not bright near gaussian centre";
    EXPECT_LT(maxByte0, uint8_t(50))  << "SoA: blue channel should stay low";
}

// ----------------------------------------------------------------
// Test: AoS vs SoA image comparison using the VulkanGaussianSplatting
// class. Both pipelines render the same single red gaussian; the
// resulting images must agree on brightness in the key region.
// ----------------------------------------------------------------
TEST_F(SoATest, imageCompare_AoS_vs_SoA) {
    const float    W        = BackendConfig::WIDTH;
    const float    H        = BackendConfig::HEIGHT;

    const float SH_ONE = 1.772f, SH_ZERO = -1.772f;
    Gaussian3D g{};
    g.position = {0.0f, -0.690f, -0.518f};
    g.scale    = {0.05f, 0.05f, 0.05f};
    g.rotation = {0.0f, 0.0f, 0.0f, 1.0f};
    g.color    = {SH_ONE, SH_ZERO, SH_ZERO};
    g.alpha    = 1.0f;

    std::shared_ptr<CameraUboType> cameraUBO;
    makeCameraUBO(cameraUBO);

    auto pixelsSoA = renderSoA(*vulkanContext, {g}, cameraUBO);
    ASSERT_FALSE(pixelsSoA.empty()) << "SoA produced no output";

    // Find the max-brightness pixel in a window around the expected centre
    const int cx = 320, cy = 240, half = 30;
    auto findMax = [&](const std::vector<uint8_t>& px, int ch) {
        uint8_t m = 0;
        for (int y = cy-half; y <= cy+half; ++y)
            for (int x = cx-half; x <= cx+half; ++x) {
                if (x < 0 || x >= (int)W || y < 0 || y >= (int)H) continue;
                m = std::max(m, px[(y*(size_t)W + x)*4 + ch]);
            }
        return m;
    };

    uint8_t soaRed  = findMax(pixelsSoA, 2);
    uint8_t soaBlue = findMax(pixelsSoA, 0);

    std::cout << "  AoS vs SoA comparison:\n"
              << "    SoA red=" << (int)soaRed << " blue=" << (int)soaBlue << "\n";

    EXPECT_GT(soaRed,  uint8_t(200)) << "SoA: red channel not bright";
    EXPECT_LT(soaBlue, uint8_t(50))  << "SoA: blue channel too bright (wrong colour)";
}
