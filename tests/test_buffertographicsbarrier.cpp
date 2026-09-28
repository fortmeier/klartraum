/**
 * TESTS:
 * - ordersComputeBufferBeforeRenderPass: wires a compute-written buffer through
 *   BufferToGraphicsBarrier into a RenderPass via addComputeDependency, compiles
 *   and submits the graph, and confirms it runs cleanly (no validation-layer
 *   errors — the debug callback throws on VK_ERROR severity) — i.e. the new
 *   compute->graphics ordering primitive and its VkBufferMemoryBarrier
 *   (DRAW_INDIRECT | VERTEX_SHADER destination stages) are structurally valid
 *   and scheduled ahead of the render pass that would consume the buffer
 **/
#include <gtest/gtest.h>

#include <vector>

#include "klartraum/headless_frontend.hpp"
#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/imageviewsrc.hpp"
#include "klartraum/computegraph/renderpass.hpp"
#include "klartraum/computegraph/buffertographicsbarrier.hpp"
#include "klartraum/computegraph/bufferelement.hpp"
#include "klartraum/draw_basics.hpp"
#include "klartraum/vulkan_buffer.hpp"

using namespace klartraum;

TEST(BufferToGraphicsBarrier, ordersComputeBufferBeforeRenderPass) {
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

    // Stand-in for a compute-produced buffer an indirect draw would consume
    // (sorted indices / VkDrawIndirectCommand args): a storage+indirect buffer
    // that gets (re-)written every frame, analogous to the dist.comp output.
    auto computedBuffer = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vc, 4,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    computedBuffer->setRecordToZero(true);

    auto barrier = std::make_shared<BufferToGraphicsBarrier>();
    barrier->addBuffer(computedBuffer);
    barrier->setInput(computedBuffer);

    auto rp = std::make_shared<RenderPass>(vc.getSwapChainImageFormat(), vc.getSwapChainExtent());
    rp->setInput(ivs, 0);
    rp->setInput(cam, 1);
    rp->addDrawComponent(std::make_shared<DrawBasics>(DrawBasicsType::Axes));
    rp->addComputeDependency(barrier);

    auto cg = ComputeGraph(vc, numImages);
    cg.compileFrom(rp);

    for (uint32_t pathId = 0; pathId < numImages; pathId++) {
        cg.submitAndWait(vc.getGraphicsQueue(), pathId);
    }
}
