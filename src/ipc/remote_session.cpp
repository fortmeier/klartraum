// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/ipc/remote_session.hpp"

#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <thread>

#include <spawn.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

#include "klartraum/computegraph/generalcomputation.hpp"
#include "klartraum/computegraph/noop.hpp"
#include "klartraum/ipc/klartraum_ipc_generated.h"
#include "klartraum/ipc/shared_buffer_element.hpp"
#include "klartraum/vulkan_helpers.hpp"

extern char** environ;

namespace klartraum::ipc {

namespace {

std::vector<uint8_t> finishRequest(::flatbuffers::FlatBufferBuilder& fbb, uint64_t requestId, Request type,
                                   ::flatbuffers::Offset<void> request) {
    auto body = CreateRequestBody(fbb, type, request);
    auto message = CreateMessage(fbb, requestId, Body::RequestBody, body.Union());
    fbb.Finish(message);
    return std::vector<uint8_t>(fbb.GetBufferPointer(), fbb.GetBufferPointer() + fbb.GetSize());
}

std::vector<uint32_t> readSpirv(const std::string& path) {
    const std::vector<char> bytes = readFile(path);
    if (bytes.size() % 4 != 0) {
        throw std::runtime_error("describeComputeGraph: " + path + " is not SPIR-V");
    }
    std::vector<uint32_t> words(bytes.size() / 4);
    std::memcpy(words.data(), bytes.data(), bytes.size());
    return words;
}

void writeAll(int fd, const uint8_t* data, size_t size) {
    while (size > 0) {
        const ssize_t written = ::write(fd, data, size);
        if (written < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw std::runtime_error(std::string("RemoteSession: write failed: ") + std::strerror(errno));
        }
        data += written;
        size -= static_cast<size_t>(written);
    }
}

void readAll(int fd, uint8_t* data, size_t size) {
    while (size > 0) {
        const ssize_t got = ::read(fd, data, size);
        if (got < 0 && errno == EINTR) {
            continue;
        }
        if (got <= 0) {
            throw std::runtime_error("RemoteSession: the server closed the connection");
        }
        data += got;
        size -= static_cast<size_t>(got);
    }
}

int connectTo(const std::string& socketPath) {
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    if (socketPath.size() >= sizeof(address.sun_path)) {
        throw std::invalid_argument("RemoteSession: socket path too long: " + socketPath);
    }
    std::strncpy(address.sun_path, socketPath.c_str(), sizeof(address.sun_path) - 1);
    const int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        throw std::runtime_error(std::string("RemoteSession: socket failed: ") + std::strerror(errno));
    }
    if (::connect(fd, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) != 0) {
        const std::string message = std::strerror(errno);
        ::close(fd);
        throw std::runtime_error("RemoteSession: cannot connect to " + socketPath + ": " + message);
    }
    return fd;
}

} // namespace

std::vector<uint8_t> describeComputeGraph(ComputeGraphElementPtr root, uint32_t graphId, uint64_t requestId) {
    // Number the reachable elements: inputs, dependencies and scratch buffers.
    std::map<ComputeGraphElement*, uint32_t> ids;
    std::vector<ComputeGraphElementPtr> elements;
    std::deque<ComputeGraphElementPtr> queue{root};
    auto visit = [&](const ComputeGraphElementPtr& element) {
        if (element && ids.emplace(element.get(), static_cast<uint32_t>(elements.size())).second) {
            elements.push_back(element);
            queue.push_back(element);
        }
    };
    visit(root);
    queue.pop_back();
    queue.push_back(root);
    while (!queue.empty()) {
        auto element = queue.front();
        queue.pop_front();
        for (const auto& [index, input] : element->getInputs()) {
            visit(input);
        }
        for (const auto& dependency : element->getDependencies()) {
            visit(dependency);
        }
        if (auto computation = std::dynamic_pointer_cast<GeneralComputationInterface>(element)) {
            for (const auto& [scratch, zero] : computation->getScratchBuffers()) {
                visit(scratch);
            }
        }
    }

    ::flatbuffers::FlatBufferBuilder fbb;
    std::vector<::flatbuffers::Offset<SharedRegion>> regions;
    std::vector<::flatbuffers::Offset<Element>> described;
    for (const auto& element : elements) {
        const uint32_t id = ids.at(element.get());
        auto name = fbb.CreateString(element->getName());

        std::vector<::flatbuffers::Offset<Input>> inputs;
        for (const auto& [index, input] : element->getInputs()) {
            inputs.push_back(CreateInput(fbb, index, ids.at(input.get()), element->getInputSlot(index)));
        }
        auto inputVector = fbb.CreateVector(inputs);
        std::vector<uint32_t> dependencyIds;
        for (const auto& dependency : element->getDependencies()) {
            dependencyIds.push_back(ids.at(dependency.get()));
        }
        auto dependencies = fbb.CreateVector(dependencyIds);

        if (auto buffer = std::dynamic_pointer_cast<SharedBufferElement>(element)) {
            const uint32_t region = static_cast<uint32_t>(regions.size());
            regions.push_back(
                CreateSharedRegionDirect(fbb, buffer->getRegion().name().c_str(), buffer->getRegion().size()));
            ElementBuilder builder(fbb);
            builder.add_id(id);
            builder.add_kind(ElementKind::Buffer);
            builder.add_name(name);
            builder.add_region(region);
            builder.add_byte_size(buffer->getBufferMemSize());
            described.push_back(builder.Finish());
        } else if (auto computation = std::dynamic_pointer_cast<GeneralComputationInterface>(element)) {
            if (computation->usesIndirectDispatch()) {
                throw std::runtime_error("describeComputeGraph: indirect dispatches are not supported remotely");
            }
            std::vector<::flatbuffers::Offset<Spirv>> shaders;
            for (const auto& path : computation->getShaderPaths()) {
                const auto words = readSpirv(path);
                shaders.push_back(CreateSpirvDirect(fbb, &words));
            }
            std::vector<::flatbuffers::Offset<PushConstants>> pushConstants;
            for (const auto& bytes : computation->getPushConstantBytes()) {
                pushConstants.push_back(CreatePushConstantsDirect(fbb, &bytes));
            }
            std::vector<::flatbuffers::Offset<Scratch>> scratch;
            for (const auto& [buffer, zero] : computation->getScratchBuffers()) {
                scratch.push_back(CreateScratch(fbb, ids.at(buffer.get()), zero));
            }
            auto shaderVector = fbb.CreateVector(shaders);
            auto pushConstantVector = fbb.CreateVector(pushConstants);
            auto scratchVector = fbb.CreateVector(scratch);
            const auto groups = computation->getGroupCounts();
            ElementBuilder builder(fbb);
            builder.add_id(id);
            builder.add_kind(ElementKind::Computation);
            builder.add_name(name);
            builder.add_shaders(shaderVector);
            builder.add_group_count_x(groups[0]);
            builder.add_group_count_y(groups[1]);
            builder.add_group_count_z(groups[2]);
            builder.add_push_constants(pushConstantVector);
            builder.add_scratch(scratchVector);
            builder.add_inputs(inputVector);
            builder.add_dependencies(dependencies);
            described.push_back(builder.Finish());
        } else if (std::dynamic_pointer_cast<NoOp>(element)) {
            ElementBuilder builder(fbb);
            builder.add_id(id);
            builder.add_kind(ElementKind::NoOp);
            builder.add_name(name);
            builder.add_inputs(inputVector);
            builder.add_dependencies(dependencies);
            described.push_back(builder.Finish());
        } else {
            throw std::runtime_error(std::string("describeComputeGraph: cannot describe a ") + element->getType() +
                                     " remotely (buffers must be SharedBufferElements)");
        }
    }
    auto regionVector = fbb.CreateVector(regions);
    auto elementVector = fbb.CreateVector(described);
    auto compile = CreateCompileGraph(fbb, graphId, regionVector, elementVector, ids.at(root.get()));
    return finishRequest(fbb, requestId, Request::CompileGraph, compile.Union());
}

RemoteSession::RemoteSession(const std::string& socketPath)
    : socketFd(connectTo(socketPath)) {}

RemoteSession::~RemoteSession() {
    if (socketFd >= 0) {
        ::close(socketFd);
    }
}

ExecutionResult RemoteSession::roundTrip(const std::vector<uint8_t>& message, uint64_t requestId) {
    if (socketFd < 0) {
        throw std::logic_error("RemoteSession: the server was shut down");
    }
    const uint32_t length = static_cast<uint32_t>(message.size());
    const uint8_t header[4] = {uint8_t(length), uint8_t(length >> 8), uint8_t(length >> 16), uint8_t(length >> 24)};
    writeAll(socketFd, header, 4);
    writeAll(socketFd, message.data(), message.size());

    uint8_t responseHeader[4];
    readAll(socketFd, responseHeader, 4);
    const uint32_t responseLength = uint32_t(responseHeader[0]) | uint32_t(responseHeader[1]) << 8 |
                                    uint32_t(responseHeader[2]) << 16 | uint32_t(responseHeader[3]) << 24;
    std::vector<uint8_t> bytes(responseLength);
    readAll(socketFd, bytes.data(), bytes.size());

    ::flatbuffers::Verifier verifier(bytes.data(), bytes.size());
    if (!VerifyMessageBuffer(verifier)) {
        throw std::runtime_error("RemoteSession: invalid response");
    }
    const Message* reply = GetMessage(bytes.data());
    const Response* response = reply->body_as_Response();
    if (response == nullptr || reply->request_id() != requestId) {
        throw std::runtime_error("RemoteSession: unexpected response");
    }
    if (!response->ok()) {
        throw std::runtime_error("RemoteSession: " + std::string(response->error() ? response->error()->str() : "?"));
    }
    return {response->gpu_ms(), response->zero_copy()};
}

uint32_t RemoteSession::compile(ComputeGraphElementPtr root) {
    const uint32_t graphId = nextGraphId++;
    const uint64_t requestId = nextRequestId++;
    zeroCopy = roundTrip(describeComputeGraph(root, graphId, requestId), requestId).zeroCopy;
    return graphId;
}

ExecutionResult RemoteSession::execute(uint32_t graphId) {
    const uint64_t requestId = nextRequestId++;
    ::flatbuffers::FlatBufferBuilder fbb;
    return roundTrip(finishRequest(fbb, requestId, Request::Execute, CreateExecute(fbb, graphId).Union()), requestId);
}

void RemoteSession::release(uint32_t graphId) {
    const uint64_t requestId = nextRequestId++;
    ::flatbuffers::FlatBufferBuilder fbb;
    roundTrip(finishRequest(fbb, requestId, Request::Release, CreateRelease(fbb, graphId).Union()), requestId);
}

void RemoteSession::shutdownServer() {
    const uint64_t requestId = nextRequestId++;
    ::flatbuffers::FlatBufferBuilder fbb;
    roundTrip(finishRequest(fbb, requestId, Request::Shutdown, CreateShutdown(fbb).Union()), requestId);
    ::close(socketFd);
    socketFd = -1;
}

RemoteServerProcess::RemoteServerProcess(const std::string& executable, const std::string& socketPath,
                                         std::chrono::milliseconds startTimeout)
    : socketPath(socketPath) {
    std::filesystem::remove(socketPath);
    std::vector<char*> argv = {const_cast<char*>(executable.c_str()), const_cast<char*>(socketPath.c_str()), nullptr};
    if (posix_spawn(&pid, executable.c_str(), nullptr, nullptr, argv.data(), environ) != 0) {
        pid = -1;
        throw std::runtime_error("RemoteServerProcess: cannot start " + executable);
    }
    // The server binds the socket once its Vulkan device is ready.
    const auto deadline = std::chrono::steady_clock::now() + startTimeout;
    while (true) {
        try {
            ::close(connectTo(socketPath));
            return;
        } catch (const std::runtime_error&) {
        }
        int status = 0;
        if (waitpid(pid, &status, WNOHANG) == pid) {
            pid = -1;
            throw std::runtime_error("RemoteServerProcess: " + executable + " ended before listening");
        }
        if (std::chrono::steady_clock::now() > deadline) {
            kill(pid, SIGKILL);
            waitpid(pid, &status, 0);
            pid = -1;
            throw std::runtime_error("RemoteServerProcess: " + executable + " did not start listening");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}

RemoteServerProcess::~RemoteServerProcess() {
    if (pid < 0) {
        return;
    }
    try {
        RemoteSession(socketPath).shutdownServer();
    } catch (const std::exception&) {
        kill(pid, SIGTERM);
    }
    int status = 0;
    for (int i = 0; i < 250 && waitpid(pid, &status, WNOHANG) == 0; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    if (waitpid(pid, &status, WNOHANG) == 0) {
        kill(pid, SIGKILL);
        waitpid(pid, &status, 0);
    }
}

std::string RemoteServerProcess::findServerExecutable() {
    if (const char* fromEnvironment = std::getenv("KLARTRAUM_IPC_SERVER"); fromEnvironment && *fromEnvironment) {
        return fromEnvironment;
    }
#ifdef KLARTRAUM_SOURCE_DIR
    for (const char* profile : {"release", "debug"}) {
        const auto candidate =
            std::filesystem::path(KLARTRAUM_SOURCE_DIR) / "rust" / "target" / profile / "klartraum_ipc_server";
        if (std::filesystem::exists(candidate)) {
            return candidate.string();
        }
    }
#endif
    return {};
}

} // namespace klartraum::ipc
