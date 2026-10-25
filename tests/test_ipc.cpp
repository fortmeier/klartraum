// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - sharedMemoryRegionIsAlignedAndZeroed: a created region has a short unique name, a size rounded
 *   up to the region alignment and zeroed contents
 * - sharedMemoryRegionIsSharedBetweenMappings: a second mapping opened by name sees the writes of
 *   the first and the other way round, and the name is gone once the creator is destroyed
 * - describeComputeGraphListsElements: describing an operator_add graph over SharedBufferElements
 *   gives a CompileGraph message with three regions, the buffers and the computation with its
 *   SPIR-V, group count and inputs, rooted at the computation
 * - describeComputeGraphRejectsOrdinaryBuffers: a graph reading an ordinary BufferElement cannot be
 *   described, since the server can only share SharedBufferElement memory
 * - remoteAddWritesIntoSharedMemory: the Rust server (klartraum_ipc_server, skipped when not built)
 *   computes A + B into the result's shared memory, reports zero copy, and reruns with new inputs
 *   written in place without recompiling
 * - remoteSlotInputsAndPushConstants: a doubling feeding an addition through a slot input gives
 *   2A + B, and a radix-sort histogram pass gets its push constants (as bytes) and scratch buffers
 * - localAndRemoteExecutionsAgree: the same C++ graph compiled locally by the C++ engine (importing
 *   the shared memory) and remotely by the Rust engine writes identical results
 * - remoteErrorsBecomeExceptions: executing an unknown graph throws with the server's message, and
 *   the session keeps working
 **/
#include <gtest/gtest.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/generalcomputation.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/ipc/klartraum_ipc_generated.h"
#include "klartraum/ipc/remote_session.hpp"
#include "klartraum/ipc/shared_buffer_element.hpp"
#include "klartraum/ipc/shared_memory.hpp"
#include "klartraum/vulkan_gaussian_splatting_types.hpp"

using namespace klartraum;
using namespace klartraum::ipc;

namespace {

// Short enough for the 104-character limit of Unix socket paths.
std::string socketPath() { return "/tmp/klartraum_ipc_test_" + std::to_string(getpid()) + ".sock"; }

std::shared_ptr<GeneralComputation<>> addGraph(VulkanContext& vc, std::shared_ptr<SharedBufferElement> a,
                                               std::shared_ptr<SharedBufferElement> b,
                                               std::shared_ptr<SharedBufferElement> r, uint32_t n) {
    auto add = std::make_shared<GeneralComputation<>>(vc, "shaders/operator_add.comp.spv");
    add->setInput(a, 0);
    add->setInput(b, 1);
    add->setInput(r, 2);
    add->setGroupCountX(n);
    add->setName("Add");
    return add;
}

void fill(SharedBufferElement& buffer, uint32_t n, float (*f)(uint32_t)) {
    float* data = buffer.data<float>();
    for (uint32_t i = 0; i < n; ++i) {
        data[i] = f(i);
    }
}

} // namespace

TEST(Ipc, sharedMemoryRegionIsAlignedAndZeroed) {
    auto a = SharedMemoryRegion::create(1000);
    auto b = SharedMemoryRegion::create(SharedMemoryRegion::kRegionAlignment + 1);
    EXPECT_NE(a.name(), b.name());
    EXPECT_LE(a.name().size(), 31u);
    EXPECT_EQ(a.size(), SharedMemoryRegion::kRegionAlignment);
    EXPECT_EQ(b.size(), 2 * SharedMemoryRegion::kRegionAlignment);
    const auto* bytes = a.as<uint8_t>();
    for (size_t i = 0; i < a.size(); ++i) {
        ASSERT_EQ(bytes[i], 0);
    }
}

TEST(Ipc, sharedMemoryRegionIsSharedBetweenMappings) {
    std::string name;
    {
        auto created = SharedMemoryRegion::create(4096);
        name = created.name();
        auto opened = SharedMemoryRegion::open(created.name(), created.size());
        created.as<float>()[10] = 3.5f;
        EXPECT_EQ(opened.as<float>()[10], 3.5f);
        opened.as<uint32_t>()[0] = 0xdeadbeef;
        EXPECT_EQ(created.as<uint32_t>()[0], 0xdeadbeefu);
    }
    EXPECT_THROW(SharedMemoryRegion::open(name, 4096), std::runtime_error);
}

TEST(Ipc, describeComputeGraphListsElements) {
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    const uint32_t n = 256;
    auto a = SharedBufferElement::create<float>(n);
    auto b = SharedBufferElement::create<float>(n);
    auto r = SharedBufferElement::create<float>(n);
    auto add = addGraph(vc, a, b, r, n);

    const auto bytes = describeComputeGraph(add, 5, 9);
    flatbuffers::Verifier verifier(bytes.data(), bytes.size());
    ASSERT_TRUE(VerifyMessageBuffer(verifier));
    const Message* message = GetMessage(bytes.data());
    EXPECT_EQ(message->request_id(), 9u);
    const CompileGraph* compile = message->body_as_RequestBody()->request_as_CompileGraph();
    ASSERT_NE(compile, nullptr);
    EXPECT_EQ(compile->graph_id(), 5u);
    ASSERT_EQ(compile->regions()->size(), 3u);
    EXPECT_EQ(compile->regions()->Get(0)->size(), SharedMemoryRegion::kRegionAlignment);
    ASSERT_EQ(compile->elements()->size(), 4u);

    const Element* root = compile->elements()->Get(compile->root());
    EXPECT_EQ(root->kind(), ElementKind::Computation);
    EXPECT_EQ(root->name()->str(), "Add");
    EXPECT_EQ(root->group_count_x(), n);
    ASSERT_EQ(root->shaders()->size(), 1u);
    EXPECT_EQ(root->shaders()->Get(0)->code()->Get(0), 0x07230203u); // SPIR-V magic
    ASSERT_EQ(root->inputs()->size(), 3u);
    for (const Input* input : *root->inputs()) {
        const Element* buffer = compile->elements()->Get(input->element());
        EXPECT_EQ(buffer->kind(), ElementKind::Buffer);
        EXPECT_EQ(buffer->byte_size(), n * sizeof(float));
        EXPECT_EQ(input->slot(), -1);
    }
}

TEST(Ipc, describeComputeGraphRejectsOrdinaryBuffers) {
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    auto a = std::make_shared<BufferElement<VulkanBuffer<float>>>(vc, 16);
    auto add = addGraph(vc, SharedBufferElement::create<float>(16), SharedBufferElement::create<float>(16),
                        SharedBufferElement::create<float>(16), 16);
    add->setInput(a, 0);
    EXPECT_THROW(describeComputeGraph(add, 1, 1), std::runtime_error);
}

class IpcRemote : public ::testing::Test {
protected:
    void SetUp() override {
        executable = RemoteServerProcess::findServerExecutable();
        if (executable.empty()) {
            GTEST_SKIP() << "klartraum_ipc_server not built; run `cargo build --release -p klartraum_ipc` in rust/";
        }
        server = std::make_unique<RemoteServerProcess>(executable, socketPath());
    }

    void TearDown() override { server.reset(); }

    std::string executable;
    std::unique_ptr<RemoteServerProcess> server;
    HeadlessFrontend frontend;
};

TEST_F(IpcRemote, remoteAddWritesIntoSharedMemory) {
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    const uint32_t n = 10000;
    auto a = SharedBufferElement::create<float>(n);
    auto b = SharedBufferElement::create<float>(n);
    auto r = SharedBufferElement::create<float>(n);
    fill(*a, n, [](uint32_t i) { return float(i); });
    fill(*b, n, [](uint32_t i) { return 0.5f; });
    auto add = addGraph(vc, a, b, r, n);

    RemoteSession session(server->getSocketPath());
    const uint32_t graph = session.compile(add);
    EXPECT_TRUE(session.isZeroCopy());

    const ExecutionResult first = session.execute(graph);
    EXPECT_TRUE(first.zeroCopy);
    for (uint32_t i = 0; i < n; ++i) {
        ASSERT_EQ(r->data<float>()[i], float(i) + 0.5f) << "at " << i;
    }

    fill(*b, n, [](uint32_t i) { return -float(i); });
    session.execute(graph);
    for (uint32_t i = 0; i < n; ++i) {
        ASSERT_EQ(r->data<float>()[i], 0.0f) << "at " << i;
    }
    std::cout << "  remote add: " << first.gpuMs << " ms GPU, zero copy " << first.zeroCopy << "\n";
    session.release(graph);
}

TEST_F(IpcRemote, remoteSlotInputsAndPushConstants) {
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    const uint32_t n = 512;
    auto a = SharedBufferElement::create<float>(n);
    auto unused = SharedBufferElement::create<float>(n);
    auto doubled = SharedBufferElement::create<float>(n);
    auto b = SharedBufferElement::create<float>(n);
    auto r = SharedBufferElement::create<float>(n);
    fill(*a, n, [](uint32_t i) { return float(i); });
    fill(*b, n, [](uint32_t i) { return 10.0f; });

    auto doubling = std::make_shared<GeneralComputation<>>(vc, "shaders/operator_double.comp.spv");
    doubling->setInput(a, 0);
    doubling->setInput(unused, 1);
    doubling->setInput(doubled, 2);
    doubling->setGroupCountX(n);
    auto add = std::make_shared<GeneralComputation<>>(vc, "shaders/operator_add.comp.spv");
    add->setInput(doubling, 0, 2); // the buffer doubling writes
    add->setInput(b, 1);
    add->setInput(r, 2);
    add->setGroupCountX(n);

    // A radix-sort histogram pass: push constants and scratch buffers, keys 0..n-1.
    auto keys = SharedBufferElement::create<uint32_t>(n);
    for (uint32_t i = 0; i < n; ++i) {
        keys->data<uint32_t>()[i] = i;
    }
    auto histogramBins = SharedBufferElement::create<uint32_t>(16);
    auto histogram = std::make_shared<RadixSort>(
        vc, std::vector<std::string>{"shaders/gsplat/gsplat_radix_sort_histogram.comp.spv"});
    histogram->setInput(keys, 0);
    histogram->setInput(SharedBufferElement::create<uint32_t>(n), 1);
    histogram->setInput(SharedBufferElement::create<uint32_t>(n), 2);
    histogram->setInput(SharedBufferElement::create<uint32_t>(n), 3);
    histogram->addScratchBufferElement(SharedBufferElement::create<uint32_t>(16), true);
    histogram->addScratchBufferElement(SharedBufferElement::create<uint32_t>(2), true);
    histogram->addScratchBufferElement(SharedBufferElement::create<uint32_t>(1), false);
    histogram->addScratchBufferElement(histogramBins, true);
    histogram->setPushConstants({SortPushConstants{0, n, 16, 0}});
    histogram->setGroupCountX(1);

    RemoteSession session(server->getSocketPath());
    session.execute(session.compile(add));
    for (uint32_t i = 0; i < n; ++i) {
        ASSERT_EQ(r->data<float>()[i], 2.0f * float(i) + 10.0f) << "at " << i;
    }
    session.execute(session.compile(histogram));
    for (uint32_t bin = 0; bin < 16; ++bin) {
        EXPECT_EQ(histogramBins->data<uint32_t>()[bin], n / 16) << "bin " << bin;
    }
}

TEST_F(IpcRemote, localAndRemoteExecutionsAgree) {
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    if (!vc.isExternalMemoryHostSupported()) {
        GTEST_SKIP() << "VK_EXT_external_memory_host is not supported";
    }
    const uint32_t n = 4096;
    auto a = SharedBufferElement::create<float>(n);
    auto b = SharedBufferElement::create<float>(n);
    auto r = SharedBufferElement::create<float>(n);
    fill(*a, n, [](uint32_t i) { return std::sin(float(i)); });
    fill(*b, n, [](uint32_t i) { return std::cos(float(i)); });
    auto add = addGraph(vc, a, b, r, n);

    ComputeGraph local(vc, 1);
    local.compileFrom(add);
    local.submitAndWait(vc.getGraphicsQueue(), 0);
    const std::vector<float> localResult(r->data<float>(), r->data<float>() + n);

    std::memset(r->data<float>(), 0, n * sizeof(float));
    RemoteSession session(server->getSocketPath());
    session.execute(session.compile(add));
    EXPECT_EQ(std::memcmp(localResult.data(), r->data<float>(), n * sizeof(float)), 0);
    EXPECT_EQ(localResult[1], std::sin(1.0f) + std::cos(1.0f));
}

TEST_F(IpcRemote, remoteErrorsBecomeExceptions) {
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    RemoteSession session(server->getSocketPath());
    try {
        session.execute(77);
        FAIL() << "executing an unknown graph did not throw";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("unknown graph 77"), std::string::npos) << e.what();
    }
    auto add = addGraph(vc, SharedBufferElement::create<float>(4), SharedBufferElement::create<float>(4),
                        SharedBufferElement::create<float>(4), 4);
    EXPECT_NO_THROW(session.execute(session.compile(add)));
}
