// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - create: builds a multi-element compute graph (RenderPass→BlurOp→NoiseOp→AddOp→CopyOp) and submits one frame
 * - headlessSubmitAndWait: DrawBasics render graph over headless paths, simple path cycling with submitAndWait
 * - headlessSubmitTo: DrawBasics render graph over headless paths using beginRender/submitTo/endRender
 * - glfwSubmitAndWait: DrawBasics render graph over GLFW paths, simple path cycling with submitAndWait
 * - glfwSubmitTo: DrawBasics render graph over GLFW paths using beginRender/submitTo/endRender
 * - executionOnlyDependenciesAreUnique: resource-order dependencies remain separate from shader inputs
 * - profilingSpansMultipleQueryPools: a graph with more elements than one timestamp pool holds reports a nonzero GPU
 * time for every dispatch
 * - longDependencyChainCompletes: a chain of thousands of dependent dispatches (larger than the SD1.5 UNet graph)
 * finishes and applies every dispatch in order
 * - submitChecksQueueFamily: a graph compiled for a queue family runs on a queue of that family, and submitting it
 * uncompiled or to a queue of another family throws
 * - backgroundQueueRunsBesideFrames: a graph compiled for and run on the background queue by a worker thread computes
 * the right result while the main thread renders frames, and the main thread reads its output
 * - elementsSetUpOnGraphQueue: compileFrom() hands each element the queue of the family it compiles for
 * (getSetupQueue()), also to the scratch inputs a GeneralComputation sets up itself
 * - immediateSubmissionsChooseQueue: submitImmediate(), copyBufferImmediate() and device-local uploads and downloads
 * run on the background queue when given it
 **/

#include <atomic>
#include <map>
#include <thread>
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
static std::pair<std::shared_ptr<RenderPass>, uint32_t> makeRenderGraph(VulkanContext& vc, bool withSemaphoreWait) {
    uint32_t numImages = vc.getNumberOfSwapChainImages();
    std::vector<VkImageView> views;
    std::vector<VkImage> imgs;
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
    auto rp = std::make_shared<RenderPass>(vc.getSwapChainImageFormat(), vc.getSwapChainExtent());
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
    std::vector<VkImage> images;
    for (int i = 0; i < 2; i++) {
        imageViews.push_back(vc.getImageView(i));
        images.push_back(vc.getSwapChainImage(i));
    }
    auto imageViewSrc = std::make_shared<ImageViewSrc>(imageViews, images);
    auto camera = std::make_shared<CameraUboType>();
    auto renderpass = std::make_shared<RenderPass>(vc.getSwapChainImageFormat(), vc.getSwapChainExtent());
    renderpass->setInput(imageViewSrc, 0);
    renderpass->setInput(camera, 1);

    auto blur = std::make_shared<BlurOp>();
    blur->setInput(renderpass);
    auto noise = std::make_shared<NoiseOp>();
    noise->setInput(blur);
    auto add = std::make_shared<AddOp>();
    add->setInput(blur, 0);
    add->setInput(noise, 1);
    auto copy = std::make_shared<CopyOp>();
    copy->setInput(add);

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
        VkSemaphore finishSem = cg.submitTo(vc.getGraphicsQueue(), imageIndex, fence);
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
        VkSemaphore finishSem = cg.submitTo(vc.getGraphicsQueue(), imageIndex, fence);
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
    for (float value : output)
        EXPECT_FLOAT_EQ(value, 3.0f);

    int timedDispatches = 0;
    for (const auto& [name, ms] : graph.getProfilingResults()) {
        if (name.rfind("chain_", 0) != 0)
            continue;
        ++timedDispatches;
        EXPECT_GT(ms, 0.0f) << name;
    }
    EXPECT_EQ(timedDispatches, kDispatches);
}

// ----------------------------------------------------------------
// Test: longDependencyChainCompletes
// A chain of dependent dispatches with more elements than the SD1.5 UNet
// graph. Each dispatch negates its input, so the final sign shows that every
// dispatch ran, in order, on the previous result.
// ----------------------------------------------------------------
TEST(ComputeGraph, longDependencyChainCompletes) {
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();

    typedef VulkanBuffer<float> FloatBuffer;
    constexpr uint32_t kLength = 64;
    constexpr int kDispatches = 3001;
    const std::string shaderPath = "shaders/operator_multiply_scalar_element_wise.comp.spv";

    auto factors = std::make_shared<BufferElement<FloatBuffer>>(vc, kLength);
    auto first = std::make_shared<BufferElement<FloatBuffer>>(vc, kLength);
    std::shared_ptr<GeneralComputation<>> last;
    for (int i = 0; i < kDispatches; ++i) {
        auto op = std::make_shared<GeneralComputation<>>(vc, shaderPath);
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
    graph.compileFrom(last);

    first->getBuffer(0).memcopyFrom(std::vector<float>(kLength, 3.0f));
    factors->getBuffer(0).memcopyFrom(std::vector<float>(kLength, -1.0f));

    for (int run = 0; run < 2; ++run) {
        graph.submitAndWait(vc.getGraphicsQueue(), 0);

        std::vector<float> output(kLength, 0.0f);
        last->getOutputElement<BufferElement<FloatBuffer>>(2)->getBuffer(0).memcopyTo(output);
        for (float value : output)
            EXPECT_FLOAT_EQ(value, -3.0f) << "run " << run;
    }
}

// ----------------------------------------------------------------
// Test: submitChecksQueueFamily
// A single dispatch compiled for the background queue family. On devices
// whose background queue has its own family (e.g. MoltenVK), submitting the
// graph to the graphics queue must throw instead of submitting command
// buffers to a queue of the wrong family.
// ----------------------------------------------------------------
TEST(ComputeGraph, submitChecksQueueFamily) {
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();

    typedef VulkanBuffer<float> FloatBuffer;
    constexpr uint32_t kLength = 64;
    const std::string shaderPath = "shaders/operator_multiply_scalar_element_wise.comp.spv";

    auto input = std::make_shared<BufferElement<FloatBuffer>>(vc, kLength);
    auto factors = std::make_shared<BufferElement<FloatBuffer>>(vc, kLength);
    auto op = std::make_shared<GeneralComputation<>>(vc, shaderPath);
    op->setInput(input, 0);
    op->setInput(factors, 1);
    op->setInput(std::make_shared<BufferElement<FloatBuffer>>(vc, kLength), 2);
    op->setGroupCountX(kLength);

    auto graph = ComputeGraph(vc, 1);
    EXPECT_THROW(graph.submitAndWait(vc.getBackgroundQueue(), 0), std::logic_error);

    graph.compileFrom(op, vc.getBackgroundQueueFamily());
    input->getBuffer(0).memcopyFrom(std::vector<float>(kLength, 3.0f));
    factors->getBuffer(0).memcopyFrom(std::vector<float>(kLength, 2.0f));
    graph.submitAndWait(vc.getBackgroundQueue(), 0);

    std::vector<float> output(kLength, 0.0f);
    op->getOutputElement<BufferElement<FloatBuffer>>(2)->getBuffer(0).memcopyTo(output);
    for (float value : output)
        EXPECT_FLOAT_EQ(value, 6.0f);

    if (vc.getBackgroundQueueFamily() != vc.getQueueFamily(vc.getGraphicsQueue())) {
        EXPECT_THROW(graph.submitAndWait(vc.getGraphicsQueue(), 0), std::invalid_argument);
    } else {
        graph.submitAndWait(vc.getGraphicsQueue(), 0);
    }
}

// ----------------------------------------------------------------
// Test: backgroundQueueRunsBesideFrames
// A worker thread builds a chain of negations, compiles it for the background
// queue family and runs it on the background queue, then copies the output
// with copyBufferImmediate on the background queue. Meanwhile the main thread
// renders frames on the graphics queue. Afterwards the main thread copies the
// worker's output again; reading the background queue's result on the
// graphics queue needs the buffers shared between the queue families.
// ----------------------------------------------------------------
TEST(ComputeGraph, backgroundQueueRunsBesideFrames) {
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();

    typedef VulkanBuffer<float> FloatBuffer;
    constexpr uint32_t kLength = 64;
    constexpr int kDispatches = 501;
    const std::string shaderPath = "shaders/operator_multiply_scalar_element_wise.comp.spv";
    constexpr VkBufferUsageFlags kCopyable =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    auto [rp, numImages] = makeRenderGraph(vc, true);
    auto frames = ComputeGraph(vc, numImages);
    frames.compileFrom(rp);

    std::shared_ptr<BufferElement<FloatBuffer>> copied;
    std::vector<std::vector<float>> workerOutputs;
    std::string workerError;
    std::atomic<bool> workerDone{false};
    std::thread worker([&] {
        try {
            auto factors = std::make_shared<BufferElement<FloatBuffer>>(vc, kLength);
            auto first = std::make_shared<BufferElement<FloatBuffer>>(vc, kLength);
            std::shared_ptr<GeneralComputation<>> last;
            for (int i = 0; i < kDispatches; ++i) {
                auto op = std::make_shared<GeneralComputation<>>(vc, shaderPath);
                if (last) {
                    op->setInput(last, 0, 2);
                } else {
                    op->setInput(first, 0);
                }
                op->setInput(factors, 1);
                op->setInput(i + 1 == kDispatches ? std::make_shared<BufferElement<FloatBuffer>>(vc, kLength, kCopyable)
                                                  : std::make_shared<BufferElement<FloatBuffer>>(vc, kLength),
                             2);
                op->setGroupCountX(kLength);
                last = op;
            }
            auto graph = ComputeGraph(vc, 1);
            graph.compileFrom(last, vc.getBackgroundQueueFamily());
            first->getBuffer(0).memcopyFrom(std::vector<float>(kLength, 3.0f));
            factors->getBuffer(0).memcopyFrom(std::vector<float>(kLength, -1.0f));

            auto& output = last->getOutputElement<BufferElement<FloatBuffer>>(2)->getBuffer(0);
            for (int run = 0; run < 3; ++run) {
                graph.submitAndWait(vc.getBackgroundQueue(), 0);
                workerOutputs.emplace_back(kLength, 0.0f);
                output.memcopyTo(workerOutputs.back());
            }
            copied = std::make_shared<BufferElement<FloatBuffer>>(vc, kLength, kCopyable);
            copied->_setup(vc, 1);
            vc.copyBufferImmediate(output.getBuffer(), copied->getBuffer(0).getBuffer(), kLength * sizeof(float),
                                   vc.getBackgroundQueue());
        } catch (const std::exception& e) {
            workerError = e.what();
        }
        workerDone = true;
    });

    int framesRendered = 0;
    while (!workerDone || framesRendered < 3) {
        auto [imageIndex, fence] = vc.beginRender();
        VkSemaphore finished = frames.submitTo(vc.getGraphicsQueue(), imageIndex, fence);
        vc.endRender(imageIndex, finished);
        ++framesRendered;
    }
    worker.join();
    vc.queueWaitIdle(vc.getGraphicsQueue());
    ASSERT_EQ(workerError, "");

    ASSERT_EQ(workerOutputs.size(), 3u);
    for (const auto& output : workerOutputs) {
        for (float value : output)
            EXPECT_FLOAT_EQ(value, -3.0f);
    }

    auto readBack = std::make_shared<BufferElement<FloatBuffer>>(vc, kLength);
    readBack->_setup(vc, 1);
    vc.copyBufferImmediate(copied->getBuffer(0).getBuffer(), readBack->getBuffer(0).getBuffer(),
                           kLength * sizeof(float));
    std::vector<float> values(kLength, 0.0f);
    readBack->getBuffer(0).memcopyTo(values);
    for (float value : values)
        EXPECT_FLOAT_EQ(value, -3.0f);
}

// ----------------------------------------------------------------
// Test: elementsSetUpOnGraphQueue
// ----------------------------------------------------------------
namespace {
// Remembers the queue it was set up for.
class SetupQueueProbe : public ComputeGraphElement {
public:
    const char* getType() const override { return "SetupQueueProbe"; }
    void checkInput(ComputeGraphElementPtr, int) override {}
    void _setup(VulkanContext& vulkanContext, uint32_t numberPaths) override {
        ComputeGraphElement::_setup(vulkanContext, numberPaths);
        setUpOn = getSetupQueue();
    }
    void _record(VkCommandBuffer, uint32_t) override {}
    VkQueue setUpOn = VK_NULL_HANDLE;
};

// A buffer that remembers the queue it was set up for.
class SetupQueueBuffer : public BufferElement<VulkanBuffer<float>> {
public:
    using BufferElement<VulkanBuffer<float>>::BufferElement;
    void _setup(VulkanContext& vulkanContext, uint32_t numberPaths) override {
        BufferElement<VulkanBuffer<float>>::_setup(vulkanContext, numberPaths);
        setUpOn = getSetupQueue();
    }
    VkQueue setUpOn = VK_NULL_HANDLE;
};
} // namespace

TEST(ComputeGraph, elementsSetUpOnGraphQueue) {
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();

    auto graphicsProbe = std::make_shared<SetupQueueProbe>();
    auto graphics = ComputeGraph(vc, 1);
    graphics.compileFrom(graphicsProbe);
    EXPECT_EQ(graphicsProbe->setUpOn, vc.getGraphicsQueue());

    auto backgroundProbe = std::make_shared<SetupQueueProbe>();
    auto background = ComputeGraph(vc, 1);
    background.compileFrom(backgroundProbe, vc.getBackgroundQueueFamily());
    EXPECT_EQ(backgroundProbe->setUpOn, vc.getQueueOfFamily(vc.getBackgroundQueueFamily()));
    if (vc.hasOwnBackgroundQueue() && vc.getBackgroundQueueFamily() != vc.getQueueFamily(vc.getGraphicsQueue())) {
        EXPECT_EQ(backgroundProbe->setUpOn, vc.getBackgroundQueue());
    }

    // A GeneralComputation sets up its scratch inputs itself and passes the queue on.
    typedef VulkanBuffer<float> FloatBuffer;
    constexpr uint32_t kLength = 16;
    auto op = std::make_shared<GeneralComputation<>>(vc, "shaders/operator_multiply_scalar_element_wise.comp.spv");
    auto scratch = std::make_shared<SetupQueueBuffer>(vc, kLength);
    op->setInput(std::make_shared<BufferElement<FloatBuffer>>(vc, kLength), 0);
    op->setInput(std::make_shared<BufferElement<FloatBuffer>>(vc, kLength), 1);
    op->setInput(std::make_shared<BufferElement<FloatBuffer>>(vc, kLength), 2);
    op->setGroupCountX(kLength);
    op->addScratchBufferElement(scratch);
    auto withScratch = ComputeGraph(vc, 1);
    withScratch.compileFrom(op, vc.getBackgroundQueueFamily());
    EXPECT_EQ(scratch->setUpOn, vc.getQueueOfFamily(vc.getBackgroundQueueFamily()));
}

// ----------------------------------------------------------------
// Test: immediateSubmissionsChooseQueue
// A device-local buffer uploaded and read back through staging copies on the
// background queue, and a fill recorded with submitImmediate on it.
// ----------------------------------------------------------------
TEST(ComputeGraph, immediateSubmissionsChooseQueue) {
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    const VkQueue background = vc.getBackgroundQueue();

    constexpr uint32_t kLength = 256;
    VulkanBuffer<float> buffer(vc, kLength,
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                   VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    std::vector<float> values(kLength);
    for (uint32_t i = 0; i < kLength; ++i)
        values[i] = static_cast<float>(i);
    buffer.memcopyFrom(values, background);
    std::vector<float> readBack(kLength, -1.0f);
    buffer.memcopyTo(readBack, background);
    EXPECT_EQ(readBack, values);

    vc.submitImmediate(
        [&](VkCommandBuffer commandBuffer) { vkCmdFillBuffer(commandBuffer, buffer.getBuffer(), 0, VK_WHOLE_SIZE, 0); },
        background);
    buffer.memcopyTo(readBack); // on the graphics queue
    for (float value : readBack)
        EXPECT_EQ(value, 0.0f);
}
