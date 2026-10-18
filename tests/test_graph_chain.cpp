// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - singleLinkMatchesSubmitAndWait: a chain of one graph computes what submitAndWait() of the graph computes
 * - repeatedGraphNeedsSimultaneousUse: adding a graph twice throws unless it was compiled after
 *   enableSimultaneousUse()
 * - repeatedGraphRunsEveryTime: a graph doubling a buffer in place, four times in one chain, multiplies it by 16
 * - linksRunInChainOrder: two graphs (double, add one) give results that depend on their order in the chain
 * - chainRunsEveryFrame: the same chain submitted for several frames, with host data reset in between, gives the
 *   same result each frame
 * - sharedBufferElementAllocatedOnce: a BufferElement in two graphs keeps the buffers of the first graph, and a graph
 *   with another number of paths cannot share it
 * - pathsInFlightStaySeparate: two chains over different paths of the same graphs, both submitted before waiting,
 *   each compute on their own path's buffers
 **/

#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "klartraum/computegraph/bufferelement.hpp"
#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/generalcomputation.hpp"
#include "klartraum/computegraph/graphchain.hpp"
#include "klartraum/headless_frontend.hpp"

using namespace klartraum;

namespace {

constexpr uint32_t kLength = 64;
using FloatBuffer = BufferElementSinglePath<VulkanBuffer<float>>;

std::shared_ptr<FloatBuffer> makeBuffer(VulkanContext& vc, float value) {
    auto buffer = std::make_shared<FloatBuffer>(vc, kLength);
    buffer->getBuffer().memcopyFrom(std::vector<float>(kLength, value));
    return buffer;
}

void fill(const std::shared_ptr<FloatBuffer>& buffer, float value) {
    buffer->getBuffer().memcopyFrom(std::vector<float>(kLength, value));
}

std::vector<float> read(const std::shared_ptr<FloatBuffer>& buffer) {
    std::vector<float> values(kLength);
    buffer->getBuffer().memcopyTo(values);
    return values;
}

// x = x * factor, in place.
std::shared_ptr<GeneralComputation<>> multiplyInPlace(VulkanContext& vc, const std::shared_ptr<FloatBuffer>& x,
                                                      const std::shared_ptr<FloatBuffer>& factor) {
    auto op = std::make_shared<GeneralComputation<>>(vc, "shaders/operator_multiply_scalar_element_wise.comp.spv");
    op->setInput(x, 0);
    op->setInput(factor, 1);
    op->setInput(x, 2);
    op->setGroupCountX(kLength);
    return op;
}

// x = x + summand, in place.
std::shared_ptr<GeneralComputation<>> addInPlace(VulkanContext& vc, const std::shared_ptr<FloatBuffer>& x,
                                                 const std::shared_ptr<FloatBuffer>& summand) {
    auto op = std::make_shared<GeneralComputation<>>(vc, "shaders/operator_add.comp.spv");
    op->setInput(x, 0);
    op->setInput(summand, 1);
    op->setInput(x, 2);
    op->setGroupCountX(kLength);
    return op;
}

void expectAll(const std::vector<float>& values, float expected) {
    for (float value : values) {
        EXPECT_FLOAT_EQ(value, expected);
    }
}

} // namespace

TEST(GraphChain, singleLinkMatchesSubmitAndWait) {
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    auto x = makeBuffer(vc, 3.0f);
    auto factor = makeBuffer(vc, 2.0f);
    ComputeGraph graph(vc, 1);
    graph.compileFrom(multiplyInPlace(vc, x, factor));

    graph.submitAndWait(vc.getGraphicsQueue(), 0);
    expectAll(read(x), 6.0f);

    GraphChain chain(vc);
    chain.add(graph);
    chain.submitAndWait(vc.getGraphicsQueue());
    expectAll(read(x), 12.0f);
}

TEST(GraphChain, repeatedGraphNeedsSimultaneousUse) {
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    auto x = makeBuffer(vc, 1.0f);
    auto factor = makeBuffer(vc, 2.0f);
    ComputeGraph plain(vc, 1);
    plain.compileFrom(multiplyInPlace(vc, x, factor));
    GraphChain chain(vc);
    chain.add(plain);
    EXPECT_THROW(chain.add(plain), std::runtime_error);

    ComputeGraph simultaneous(vc, 1);
    simultaneous.enableSimultaneousUse();
    simultaneous.compileFrom(multiplyInPlace(vc, x, factor));
    EXPECT_TRUE(simultaneous.allowsSimultaneousUse());
    GraphChain repeated(vc);
    repeated.add(simultaneous);
    EXPECT_NO_THROW(repeated.add(simultaneous));
}

TEST(GraphChain, repeatedGraphRunsEveryTime) {
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    auto x = makeBuffer(vc, 1.0f);
    auto factor = makeBuffer(vc, 2.0f);
    ComputeGraph graph(vc, 1);
    graph.enableSimultaneousUse();
    graph.compileFrom(multiplyInPlace(vc, x, factor));

    GraphChain chain(vc);
    for (int i = 0; i < 4; ++i) {
        chain.add(graph);
    }
    chain.submitAndWait(vc.getGraphicsQueue());
    expectAll(read(x), 16.0f);
}

TEST(GraphChain, linksRunInChainOrder) {
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    auto x = makeBuffer(vc, 1.0f);
    auto two = makeBuffer(vc, 2.0f);
    auto one = makeBuffer(vc, 1.0f);
    ComputeGraph doubling(vc, 1);
    doubling.enableSimultaneousUse();
    doubling.compileFrom(multiplyInPlace(vc, x, two));
    ComputeGraph increment(vc, 1);
    increment.enableSimultaneousUse();
    increment.compileFrom(addInPlace(vc, x, one));

    GraphChain doubleFirst(vc);
    doubleFirst.add(doubling);
    doubleFirst.add(increment);
    doubleFirst.add(doubling);
    doubleFirst.submitAndWait(vc.getGraphicsQueue());
    expectAll(read(x), 6.0f); // (1 * 2 + 1) * 2

    fill(x, 1.0f);
    GraphChain incrementFirst(vc);
    incrementFirst.add(increment);
    incrementFirst.add(doubling);
    incrementFirst.add(doubling);
    incrementFirst.submitAndWait(vc.getGraphicsQueue());
    expectAll(read(x), 8.0f); // (1 + 1) * 2 * 2
}

TEST(GraphChain, chainRunsEveryFrame) {
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    auto x = makeBuffer(vc, 1.0f);
    auto two = makeBuffer(vc, 2.0f);
    auto one = makeBuffer(vc, 1.0f);
    ComputeGraph doubling(vc, 1);
    doubling.enableSimultaneousUse();
    doubling.compileFrom(multiplyInPlace(vc, x, two));
    ComputeGraph increment(vc, 1);
    increment.compileFrom(addInPlace(vc, x, one));

    GraphChain chain(vc);
    chain.add(doubling);
    chain.add(increment);
    chain.add(doubling);
    for (int frame = 0; frame < 5; ++frame) {
        fill(x, float(frame));
        chain.submitAndWait(vc.getGraphicsQueue());
        expectAll(read(x), (float(frame) * 2.0f + 1.0f) * 2.0f);
    }
}

namespace {

using PathFloats = BufferElement<VulkanBuffer<float>>;

// x = x * factor, in place, on buffers with one allocation per path.
std::shared_ptr<GeneralComputation<>> multiplyPaths(VulkanContext& vc, const std::shared_ptr<PathFloats>& x,
                                                    const std::shared_ptr<FloatBuffer>& factor) {
    auto op = std::make_shared<GeneralComputation<>>(vc, "shaders/operator_multiply_scalar_element_wise.comp.spv");
    op->setInput(x, 0);
    op->setInput(factor, 1);
    op->setInput(x, 2);
    op->setGroupCountX(kLength);
    return op;
}

} // namespace

TEST(GraphChain, sharedBufferElementAllocatedOnce) {
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    auto x = std::make_shared<PathFloats>(vc, kLength);
    auto factor = makeBuffer(vc, 2.0f);
    ComputeGraph first(vc, 2);
    first.compileFrom(multiplyPaths(vc, x, factor));
    const VkBuffer path0 = x->getVkBuffer(0);
    const VkBuffer path1 = x->getVkBuffer(1);
    EXPECT_NE(path0, path1);

    ComputeGraph second(vc, 2);
    second.compileFrom(multiplyPaths(vc, x, factor));
    EXPECT_EQ(x->getVkBuffer(0), path0);
    EXPECT_EQ(x->getVkBuffer(1), path1);

    ComputeGraph otherPaths(vc, 3);
    EXPECT_THROW(otherPaths.compileFrom(multiplyPaths(vc, x, factor)), std::runtime_error);
}

TEST(GraphChain, pathsInFlightStaySeparate) {
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    auto x = std::make_shared<PathFloats>(vc, kLength);
    auto two = makeBuffer(vc, 2.0f);
    auto three = makeBuffer(vc, 3.0f);
    ComputeGraph doubling(vc, 2);
    doubling.enableSimultaneousUse();
    doubling.compileFrom(multiplyPaths(vc, x, two));
    ComputeGraph tripling(vc, 2);
    tripling.compileFrom(multiplyPaths(vc, x, three));

    GraphChain frame0(vc), frame1(vc);
    for (auto [chain, path] : {std::pair{&frame0, 0u}, std::pair{&frame1, 1u}}) {
        chain->add(doubling, path);
        chain->add(tripling, path);
        chain->add(doubling, path);
    }
    for (int round = 0; round < 3; ++round) {
        x->getBuffer(0).memcopyFrom(std::vector<float>(kLength, 1.0f));
        x->getBuffer(1).memcopyFrom(std::vector<float>(kLength, 10.0f));
        frame0.submit(vc.getGraphicsQueue());
        frame1.submit(vc.getGraphicsQueue());
        frame0.wait();
        frame1.wait();
        std::vector<float> values(kLength);
        x->getBuffer(0).memcopyTo(values);
        expectAll(values, 12.0f);
        x->getBuffer(1).memcopyTo(values);
        expectAll(values, 120.0f);
    }
}
