// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/ipc/shared_memory.hpp"

#include <atomic>
#include <cerrno>
#include <cstring>
#include <stdexcept>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace klartraum::ipc {

namespace {

std::string errnoMessage(const std::string& what) { return what + ": " + std::strerror(errno); }

} // namespace

SharedMemoryRegion SharedMemoryRegion::create(size_t size) {
    static std::atomic<uint32_t> counter{0};
    const size_t aligned = ((size == 0 ? 1 : size) + kRegionAlignment - 1) / kRegionAlignment * kRegionAlignment;
    // "/kt" + pid + "_" + counter stays within the 31 characters macOS allows.
    const std::string name = "/kt" + std::to_string(getpid()) + "_" + std::to_string(counter++);

    const int fd = shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
    if (fd < 0) {
        throw std::runtime_error(errnoMessage("shm_open(" + name + ")"));
    }
    if (ftruncate(fd, static_cast<off_t>(aligned)) != 0) {
        const std::string message = errnoMessage("ftruncate(" + name + ")");
        close(fd);
        shm_unlink(name.c_str());
        throw std::runtime_error(message);
    }
    void* mapping = mmap(nullptr, aligned, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (mapping == MAP_FAILED) {
        const std::string message = errnoMessage("mmap(" + name + ")");
        shm_unlink(name.c_str());
        throw std::runtime_error(message);
    }
    return SharedMemoryRegion(name, mapping, aligned, true);
}

SharedMemoryRegion SharedMemoryRegion::open(const std::string& name, size_t size) {
    const int fd = shm_open(name.c_str(), O_RDWR, 0600);
    if (fd < 0) {
        throw std::runtime_error(errnoMessage("shm_open(" + name + ")"));
    }
    void* mapping = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (mapping == MAP_FAILED) {
        throw std::runtime_error(errnoMessage("mmap(" + name + ")"));
    }
    return SharedMemoryRegion(name, mapping, size, false);
}

SharedMemoryRegion::~SharedMemoryRegion() { release(); }

SharedMemoryRegion::SharedMemoryRegion(SharedMemoryRegion&& other) noexcept
    : regionName(std::move(other.regionName)),
      mapping(other.mapping),
      mappedSize(other.mappedSize),
      owner(other.owner) {
    other.mapping = nullptr;
    other.mappedSize = 0;
    other.owner = false;
}

SharedMemoryRegion& SharedMemoryRegion::operator=(SharedMemoryRegion&& other) noexcept {
    if (this != &other) {
        release();
        regionName = std::move(other.regionName);
        mapping = other.mapping;
        mappedSize = other.mappedSize;
        owner = other.owner;
        other.mapping = nullptr;
        other.mappedSize = 0;
        other.owner = false;
    }
    return *this;
}

void SharedMemoryRegion::release() {
    if (mapping != nullptr) {
        munmap(mapping, mappedSize);
        mapping = nullptr;
    }
    if (owner) {
        shm_unlink(regionName.c_str());
        owner = false;
    }
}

} // namespace klartraum::ipc
