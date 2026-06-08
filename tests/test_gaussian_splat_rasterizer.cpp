/**
 * TESTS:
 * - rendersIndirectInstancedQuadsWithoutValidationErrors: wires GaussianSplatRasterizer
 *   into a RenderPass with stand-in per-splat SoA storage buffers (positions,
 *   premultiplied colorsAlpha, sorted indices) and a seeded VkDrawIndirectCommand
 *   buffer, all scheduled ahead of the render pass via BufferToGraphicsBarrier,
 *   compiles and submits the graph across all swapchain paths, and confirms it
 *   runs cleanly (no validation-layer errors — the debug callback throws on
 *   VK_ERROR severity) — i.e. the descriptor sets (camera UBO @ set 0, SSBOs @
 *   set 1), push constants, blend-enabled no-vertex-buffer pipeline, and
 *   vkCmdDrawIndirect are structurally valid and correctly wired
 **/
#include <gtest/gtest.h>

#include <cstddef>
#include <vector>
#include <glm/glm.hpp>

#include "klartraum/headless_frontend.hpp"
#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/imageviewsrc.hpp"
#include "klartraum/computegraph/renderpass.hpp"
#include "klartraum/computegraph/buffertographicsbarrier.hpp"
#include "klartraum/computegraph/bufferelement.hpp"
#include "klartraum/computegraph/buffertransformation.hpp"
#include "klartraum/draw_basics.hpp"
#include "klartraum/gaussian_splat_rasterizer.hpp"
#include "klartraum/vulkan_buffer.hpp"

using namespace klartraum;

TEST(GaussianSplatRasterizer, rendersIndirectInstancedQuadsWithoutValidationErrors) {
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();

    uint32_t numImages = vc.getNumberOfSwapChainImages();
    std::vector<VkImageView> views;
    std::vector<VkImage> imgs;
    for (uint32_t i = 0; i < numImages; i++) {
        views.push_back(vc.getImageView(i));
        imgs.push_back(vc.getSwapChainImage(i));
    }
    auto ivs = std::make_shared<ImageViewSrc>(views, imgs);
    auto cam = std::make_shared<CameraUboType>();

    const uint32_t numSplats = 3;

    // Stand-ins for the compute culling/sort stage's outputs (guide §5 Stage A/B):
    // per-splat SoA attribute buffers, the sorted-index permutation, and the
    // VkDrawIndirectCommand args buffer (vertexCount=4, instanceCount=visible count).
    auto positions = std::make_shared<BufferElement<VulkanBuffer<glm::vec3>>>(vc, numSplats,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    auto colorsAlpha = std::make_shared<BufferElement<VulkanBuffer<glm::vec4>>>(vc, numSplats,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    auto sortedIndices = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vc, numSplats,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    auto drawArgs = std::make_shared<DrawIndirectCommandBufferElement>(vc, 1,
        VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    auto barrier = std::make_shared<BufferToGraphicsBarrier>();
    barrier->addBuffer(positions);
    barrier->addBuffer(colorsAlpha);
    barrier->addBuffer(sortedIndices);
    barrier->addBuffer(drawArgs);
    barrier->setInput(positions, 0);
    barrier->setInput(colorsAlpha, 1);
    barrier->setInput(sortedIndices, 2);
    barrier->setInput(drawArgs, 3);

    auto rp = std::make_shared<RenderPass>(vc.getSwapChainImageFormat(), vc.getSwapChainExtent());
    rp->setInput(ivs, 0);
    rp->setInput(cam, 1);
    rp->addDrawComponent(std::make_shared<DrawBasics>(DrawBasicsType::Axes));

    auto rasterizer = std::make_shared<GaussianSplatRasterizer>(
        std::vector<std::shared_ptr<BufferElementInterface>>{ positions, colorsAlpha, sortedIndices },
        drawArgs);

    auto extent = vc.getSwapChainExtent();
    GaussianSplatRasterPushConstants pushConstants{};
    pushConstants.resolution = glm::vec2((float)extent.width, (float)extent.height);
    pushConstants.focal = glm::vec2(1000.0f, 1000.0f);
    pushConstants.splatScale = 50.0f;
    pushConstants.shDegree = 0;
    rasterizer->setPushConstants(pushConstants);

    rp->addDrawComponent(rasterizer);
    rp->addComputeDependency(barrier);

    auto cg = ComputeGraph(vc, numImages);
    cg.compileFrom(rp);

    std::vector<glm::vec3> positionData = {
        {0.0f, 0.0f, -2.0f},
        {0.5f, 0.0f, -2.5f},
        {-0.5f, 0.3f, -3.0f}
    };
    std::vector<glm::vec4> colorAlphaData = {
        {1.0f, 0.0f, 0.0f, 1.0f},
        {0.0f, 1.0f, 0.0f, 0.8f},
        {0.0f, 0.0f, 1.0f, 0.6f}
    };
    std::vector<uint32_t> sortedIndexData = {0, 1, 2};
    VkDrawIndirectCommand drawArgsData{4, numSplats, 0, 0};

    for (uint32_t pathId = 0; pathId < numImages; pathId++) {
        positions->getBuffer(pathId).memcopyFrom(positionData.data(), positionData.size());
        colorsAlpha->getBuffer(pathId).memcopyFrom(colorAlphaData.data(), colorAlphaData.size());
        sortedIndices->getBuffer(pathId).memcopyFrom(sortedIndexData.data(), sortedIndexData.size());
        drawArgs->getBuffer(pathId).memcopyFrom(&drawArgsData, 1);
    }

    for (uint32_t pathId = 0; pathId < numImages; pathId++) {
        cg.submitAndWait(vc.getGraphicsQueue(), pathId);
    }
}
