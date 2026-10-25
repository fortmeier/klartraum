// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_IPC_REMOTE_SESSION_HPP
#define KLARTRAUM_IPC_REMOTE_SESSION_HPP

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include <sys/types.h>

#include "klartraum/computegraph/computegraphelement.hpp"

namespace klartraum::ipc {

/**
 * @brief Describes the compute graph `root` reaches as a CompileGraph message (schema ipc/klartraum_ipc.fbs).
 *
 * Supported elements: SharedBufferElement (as a shared-memory region), every
 * GeneralComputation (its shaders read with readFile() and sent as SPIR-V, group
 * counts, push constants as bytes, scratch buffers) and NoOp, with their inputs
 * (including slots) and execution-only dependencies. Other elements, ordinary
 * buffers among them, throw std::runtime_error: the server can only share
 * SharedBufferElement memory.
 */
std::vector<uint8_t> describeComputeGraph(ComputeGraphElementPtr root, uint32_t graphId, uint64_t requestId);

/// What an execution on the server reported.
struct ExecutionResult {
    /// GPU time of the graph in milliseconds.
    float gpuMs = 0.0f;
    /// Whether the server worked on the shared memory itself (else it copied).
    bool zeroCopy = false;
};

/**
 * @brief A connection to a klartraum_ipc_server (the Rust engine) that compiles and runs graphs remotely.
 *
 * compile() sends the description of a graph made of SharedBufferElements and
 * GeneralComputations; execute() runs it and returns when it has finished, after
 * which the results are in the SharedBufferElements' data(). Write new inputs
 * into data() and call execute() again to rerun without recompiling. The
 * SharedBufferElements must live as long as the graph is used.
 */
class RemoteSession {
public:
    /// Connects to the server's Unix socket.
    explicit RemoteSession(const std::string& socketPath);
    ~RemoteSession();

    RemoteSession(const RemoteSession&) = delete;
    RemoteSession& operator=(const RemoteSession&) = delete;

    /// Compiles the graph `root` reaches on the server and returns its id.
    uint32_t compile(ComputeGraphElementPtr root);

    /// Runs a compiled graph once and waits for it.
    ExecutionResult execute(uint32_t graphId);

    /// Frees a compiled graph on the server.
    void release(uint32_t graphId);

    /// Ends the server process; the session cannot be used afterwards.
    void shutdownServer();

    /// Whether the server shares the buffers' memory (reported by the last compile()).
    bool isZeroCopy() const { return zeroCopy; }

private:
    /// Sends a framed request and returns the response's zero_copy and gpu_ms; throws if it failed.
    ExecutionResult roundTrip(const std::vector<uint8_t>& message, uint64_t requestId);

    /// The connected socket.
    int socketFd = -1;
    uint64_t nextRequestId = 1;
    uint32_t nextGraphId = 1;
    bool zeroCopy = false;
};

/**
 * @brief Runs a klartraum_ipc_server as a child process for as long as the object lives.
 *
 * The constructor starts `executable socketPath` and waits until the socket
 * accepts connections; the destructor asks the server to shut down and waits for
 * it (killing it if it does not end).
 */
class RemoteServerProcess {
public:
    RemoteServerProcess(const std::string& executable, const std::string& socketPath,
                        std::chrono::milliseconds startTimeout = std::chrono::seconds(30));
    ~RemoteServerProcess();

    RemoteServerProcess(const RemoteServerProcess&) = delete;
    RemoteServerProcess& operator=(const RemoteServerProcess&) = delete;

    const std::string& getSocketPath() const { return socketPath; }

    /**
     * @brief The server binary: the KLARTRAUM_IPC_SERVER environment variable, else the
     *        release or debug build in the source tree's rust/target, or empty if none exists.
     */
    static std::string findServerExecutable();

private:
    std::string socketPath;
    /// The child process, or -1 once it ended.
    pid_t pid = -1;
};

} // namespace klartraum::ipc

#endif // KLARTRAUM_IPC_REMOTE_SESSION_HPP
