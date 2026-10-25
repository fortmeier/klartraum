// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Builds a small compute graph in C++ (C = 2A + B), runs it in the Rust engine
// through klartraum_ipc_server and reads the result from shared memory, without
// copying the data between the processes.

#include <cmath>
#include <iostream>
#include <string>

#include <unistd.h>

#include "klartraum/computegraph/generalcomputation.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/ipc/remote_session.hpp"
#include "klartraum/ipc/shared_buffer_element.hpp"

using namespace klartraum;

int main(int argc, char** argv) {
    // Parse args:  --server PATH   (the klartraum_ipc_server binary; default: found in rust/target)
    //              --size N        (number of floats, default 1000000)
    //              --runs N        (executions after one compilation, default 3)
    std::string serverPath = ipc::RemoteServerProcess::findServerExecutable();
    uint32_t n = 1000000;
    int runs = 3;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--server" && i + 1 < argc) {
            serverPath = argv[++i];
        } else if (arg == "--size" && i + 1 < argc) {
            n = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--runs" && i + 1 < argc) {
            runs = std::stoi(argv[++i]);
        }
    }
    if (serverPath.empty()) {
        std::cerr << "klartraum_ipc_server not found: build it with `cargo build --release -p klartraum_ipc` in rust/, "
                     "or pass --server PATH"
                  << std::endl;
        return 1;
    }

    // The C++ engine only describes the graph here; its context provides the GeneralComputation elements.
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();

    auto a = ipc::SharedBufferElement::create<float>(n);
    auto unused = ipc::SharedBufferElement::create<float>(n);
    auto doubled = ipc::SharedBufferElement::create<float>(n);
    auto b = ipc::SharedBufferElement::create<float>(n);
    auto c = ipc::SharedBufferElement::create<float>(n);

    auto doubling = std::make_shared<GeneralComputation<>>(vc, "shaders/operator_double.comp.spv");
    doubling->setName("Double");
    doubling->setInput(a, 0);
    doubling->setInput(unused, 1);
    doubling->setInput(doubled, 2);
    doubling->setGroupCountX(n);
    auto add = std::make_shared<GeneralComputation<>>(vc, "shaders/operator_add.comp.spv");
    add->setName("Add");
    add->setInput(doubling, 0, 2); // the buffer the doubling writes
    add->setInput(b, 1);
    add->setInput(c, 2);
    add->setGroupCountX(n);

    const std::string socketPath = "/tmp/klartraum_ipc_example_" + std::to_string(getpid()) + ".sock";
    ipc::RemoteServerProcess server(serverPath, socketPath);
    ipc::RemoteSession session(socketPath);
    const uint32_t graph = session.compile(add);
    std::cout << "Compiled the graph in the Rust engine (zero copy: " << (session.isZeroCopy() ? "yes" : "no") << ")"
              << std::endl;

    for (int run = 0; run < runs; ++run) {
        // Inputs are written straight into the shared memory the remote shaders read.
        float* av = a->data<float>();
        float* bv = b->data<float>();
        for (uint32_t i = 0; i < n; ++i) {
            av[i] = float(i % 1000) * 0.001f + float(run);
            bv[i] = 1.0f;
        }
        const ipc::ExecutionResult result = session.execute(graph);

        const float* cv = c->data<float>();
        uint32_t wrong = 0;
        for (uint32_t i = 0; i < n; ++i) {
            if (std::abs(cv[i] - (2.0f * av[i] + bv[i])) > 1e-5f) {
                ++wrong;
            }
        }
        std::cout << "Run " << run << ": " << result.gpuMs << " ms on the GPU, C[1] = " << cv[1] << ", " << wrong
                  << " wrong values" << std::endl;
        if (wrong != 0) {
            return 1;
        }
    }
    session.release(graph);
    return 0;
}
