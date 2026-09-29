/**
 * TESTS:
 * - create: builds a multi-element compute graph (RenderPass→BlurOp→NoiseOp→AddOp→CopyOp) and submits one frame
 * - headlessSubmitAndWait: DrawBasics render graph over headless paths, simple path cycling with submitAndWait
 * - headlessSubmitTo: DrawBasics render graph over headless paths using beginRender/submitTo/endRender
 * - glfwSubmitAndWait: DrawBasics render graph over GLFW paths, simple path cycling with submitAndWait
 * - glfwSubmitTo: DrawBasics render graph over GLFW paths using beginRender/submitTo/endRender
 * - executionOnlyDependenciesAreUnique: resource-order dependencies remain separate from shader inputs
 * - profilingSpansMultipleQueryPools: a graph with more elements than one timestamp pool holds reports a nonzero GPU time for every dispatch
 **/

#include <map>
#include <vector>

#include <gtest/gtest.h>

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/generalcomputation.hpp"
#include "klartraum/computegraph/imageviewsrc.hpp"
#include "klartraum/computegraph/renderpass.hpp"
#include "klartraum/draw_basics.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/glfw_frontend.hpp"

using namespace klartraum;

class BlurOp : public ComputeGraphElement {
    virtual const char* getType() const { return "BlurOp"; }
    virtual void checkInput(ComputeGraphElementPtr input, int index = 0) {}
    virtual void _record(VkCommandBuffer commandBuffer) {};
};

class NoiseOp : public ComputeGraphElement {
    virtual const char* getType() const { return "NoiseOp"; }
    virtual void checkInput(ComputeGraphElementPtr input, int index = 0) {}
    virtual void _record(VkCommandBuffer commandBuffer) {};
};

class AddOp : public ComputeGraphElement {
    virtual const char* getType() const { return "AddOp"; }
    virtual void checkInput(ComputeGraphElementPtr input, int index = 0) {}
    virtual void _record(VkCommandBuffer commandBuffer) {};
};

class CopyOp : public ComputeGraphElement {
    virtual const char* getType() const { return "CopyOp"; }
    virtual void checkInput(ComputeGraphElementPtr input, int index = 0) {}
    virtual void _record(VkCommandBuffer commandBuffer) {};
};

TEST(ComputeGraph, executionOnlyDependenciesAreUnique) {
    auto producer = std::make_shared<BlurOp>();
    auto consumer = std::make_shared<NoiseOp>();
    consumer->addDependency(producer);
    consumer->addDependency(producer);

    ASSERT_EQ(consumer->getDependencies().size(), 1);
    EXPECT_EQ(consumer->getDependencies().front(), producer);
    EXPECT_TRUE(consumer->getInputs().empty());
    EXPECT_THROW(consumer->addDependency(nullptr), std::invalid_argument);
    EXPECT_THROW(consumer->addDependency(consumer), std::invalid_argument);
}

// Build a DrawBasics axes render graph from a VulkanContext.
// withSemaphoreWait=true: sets imageAvailableSemaphoresPerImage on each path so
//   the graph waits for beginRender() to signal the image before rendering.
//   Required when using submitTo + beginRender/endRender.
// withSemaphoreWait=false: no image semaphore wait; suitable for submitAndWait.
static std::pair<std::shared_ptr<RenderPass>, uint32_t>
makeRenderGraph(VulkanContext& vc, bool withSemaphoreWait) {
    uint32_t numImages = vc.getNumberOfSwapChainImages();
    std::vector<VkImageView> views;
    std::vector<VkImage>     imgs;
    for (uint32_t i = 0; i < numImages; i++) {
        views.push_back(vc.getImageView(i));
        imgs.push_back(vc.getSwapChainImage(i));
    }
    auto ivs = std::make_shared<ImageViewSrc>(views, imgs);
    if (withSemaphoreWait) {
        for (uint32_t i = 0; i < numImages; i++)
            ivs->setWaitFor(i, vc.imageAvailableSemaphoresPerImage[i]);
    }
    auto cam = std::make_shared<CameraUboType>();
    auto rp  = std::make_shared<RenderPass>(vc.getSwapChainImageFormat(), vc.getSwapChainExtent());
    rp->setInput(ivs, 0);
    rp->setInput(cam, 1);
    rp->addDrawComponent(std::make_shared<DrawBasics>(DrawBasicsType::Axes));
    return {rp, numImages};
}

// ----------------------------------------------------------------
// Test: create
// Multi-element graph: RenderPass → BlurOp → NoiseOp → AddOp → CopyOp.
// Single path, single submit.
// ----------------------------------------------------------------
TEST(ComputeGraph, create) {
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();

    std::vector<VkImageView> imageViews;
    std::vector<VkImage>     images;
    for (int i = 0; i < 2; i++) {
        imageViews.push_back(vc.getImageView(i));
        images.push_back(vc.getSwapChainImage(i));
    }
    auto imageViewSrc = std::make_shared<ImageViewSrc>(imageViews, images);
    auto camera       = std::make_shared<CameraUboType>();
    auto renderpass   = std::make_shared<RenderPass>(vc.getSwapChainImageFormat(), vc.getSwapChainExtent());
    renderpass->setInput(imageViewSrc, 0);
    renderpass->setInput(camera, 1);

    auto blur  = std::make_shared<BlurOp>();  blur->setInput(renderpass);
    auto noise = std::make_shared<NoiseOp>(); noise->setInput(blur);
    auto add   = std::make_shared<AddOp>();
    add->setInput(blur, 0); add->setInput(noise, 1);
    auto copy  = std::make_shared<CopyOp>(); copy->setInput(add);

    auto cg = ComputeGraph(vc, 1);
    cg.compileFrom(copy);
    cg.submitAndWait(vc.getGraphicsQueue(), 0);
}

// ----------------------------------------------------------------
// Test: headlessSubmitAndWait
// No beginRender/endRender needed — submitAndWait handles its own sync.
// ----------------------------------------------------------------
TEST(ComputeGraph, headlessSubmitAndWait) {
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();

    auto [rp, numImages] = makeRenderGraph(vc, false);
    auto cg = ComputeGraph(vc, numImages);
    cg.compileFrom(rp);

    for (int round = 0; round < 3; round++)
        for (uint32_t pathId = 0; pathId < numImages; pathId++)
            cg.submitAndWait(vc.getGraphicsQueue(), pathId);
}

// ----------------------------------------------------------------
// Test: headlessSubmitTo
// Uses beginRender/submitTo/endRender — the standard production pattern.
// beginRender signals imageAvailableSemaphoresPerImage[imageIndex] (headless:
// cycles images without vkAcquireNextImageKHR).
// endRender drains the finish semaphore (headless: drain submit instead of
// vkQueuePresentKHR).
// ----------------------------------------------------------------
TEST(ComputeGraph, headlessSubmitTo) {
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();

    auto [rp, numImages] = makeRenderGraph(vc, true);
    auto cg = ComputeGraph(vc, numImages);
    cg.compileFrom(rp);

    for (int i = 0; i < 6; i++) {
        auto [imageIndex, fence] = vc.beginRender();
        VkSemaphore finishSem    = cg.submitTo(vc.getGraphicsQueue(), imageIndex, fence);
        vc.endRender(imageIndex, finishSem);
    }
    vkQueueWaitIdle(vc.getGraphicsQueue());
}

// ----------------------------------------------------------------
// Test: glfwSubmitAndWait
// ----------------------------------------------------------------
TEST(ComputeGraph, glfwSubmitAndWait) {
    GlfwFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();

    auto [rp, numImages] = makeRenderGraph(vc, false);
    auto cg = ComputeGraph(vc, numImages);
    cg.compileFrom(rp);

    for (int round = 0; round < 3; round++)
        for (uint32_t pathId = 0; pathId < numImages; pathId++)
            cg.submitAndWait(vc.getGraphicsQueue(), pathId);
}

// ----------------------------------------------------------------
// Test: glfwSubmitTo
// Same beginRender/submitTo/endRender pattern as headlessSubmitTo.
// GlfwFrontend currently uses offscreen images so the same branches
// in beginRender/endRender fire as in headless mode.
// ----------------------------------------------------------------
TEST(ComputeGraph, glfwSubmitTo) {
    GlfwFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();

    auto [rp, numImages] = makeRenderGraph(vc, true);
    auto cg = ComputeGraph(vc, numImages);
    cg.compileFrom(rp);

    for (int i = 0; i < 6; i++) {
        auto [imageIndex, fence] = vc.beginRender();
        VkSemaphore finishSem    = cg.submitTo(vc.getGraphicsQueue(), imageIndex, fence);
        vc.endRender(imageIndex, finishSem);
    }
    vkQueueWaitIdle(vc.getGraphicsQueue());
}

// ----------------------------------------------------------------
// Test: profilingSpansMultipleQueryPools
// A chain of dispatches whose element count (dispatches plus their output
// buffers) exceeds one timestamp query pool. Every dispatch, including those
// recorded into later pools, must report a positive GPU time.
// ----------------------------------------------------------------
TEST(ComputeGraph, profilingSpansMultipleQueryPools) {
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();

    typedef VulkanBuffer<float> FloatBuffer;
    constexpr uint32_t kLength = 64;
    constexpr int kDispatches = 1100;
    const std::string shaderPath = "shaders/operator_multiply_scalar_element_wise.comp.spv";

    auto factors = std::make_shared<BufferElement<FloatBuffer>>(vc, kLength);
    auto first = std::make_shared<BufferElement<FloatBuffer>>(vc, kLength);
    std::shared_ptr<GeneralComputation<>> last;
    for (int i = 0; i < kDispatches; ++i) {
        auto op = std::make_shared<GeneralComputation<>>(vc, shaderPath);
        op->setName("chain_" + std::to_string(i));
        if (last) {
            op->setInput(last, 0, 2);
        } else {
            op->setInput(first, 0);
        }
        op->setInput(factors, 1);
        op->setInput(std::make_shared<BufferElement<FloatBuffer>>(vc, kLength), 2);
        op->setGroupCountX(kLength);
        last = op;
    }

    auto graph = ComputeGraph(vc, 1);
    graph.enableProfiling();
    graph.compileFrom(last);

    first->getBuffer(0).memcopyFrom(std::vector<float>(kLength, 3.0f));
    factors->getBuffer(0).memcopyFrom(std::vector<float>(kLength, 1.0f));
    graph.submitAndWait(vc.getGraphicsQueue(), 0);

    std::vector<float> output(kLength, 0.0f);
    last->getOutputElement<BufferElement<FloatBuffer>>(2)->getBuffer(0).memcopyTo(output);
    for (float value : output) EXPECT_FLOAT_EQ(value, 3.0f);

    int timedDispatches = 0;
    for (const auto& [name, ms] : graph.getProfilingResults()) {
        if (name.rfind("chain_", 0) != 0) continue;
        ++timedDispatches;
        EXPECT_GT(ms, 0.0f) << name;
    }
    EXPECT_EQ(timedDispatches, kDispatches);
}
