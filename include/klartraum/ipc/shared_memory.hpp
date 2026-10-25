// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_IPC_SHARED_MEMORY_HPP
#define KLARTRAUM_IPC_SHARED_MEMORY_HPP

#include <cstddef>
#include <cstdint>
#include <string>

namespace klartraum::ipc {

/**
 * @brief A POSIX shared-memory object, mapped into this process.
 *
 * Another process maps the same pages by opening the object's name(), so data
 * written here is visible there without a copy, and vice versa. The size is
 * rounded up to kRegionAlignment, the largest host-pointer import alignment of
 * the GPUs the Rust server imports such regions on (16 KiB on Apple GPUs), so a
 * region can back a Vulkan buffer directly.
 *
 * The creating region owns the name and unlinks it on destruction; regions
 * opened with open() only unmap.
 */
class SharedMemoryRegion {
public:
    /// Alignment of region sizes in bytes.
    static constexpr size_t kRegionAlignment = 64 * 1024;

    /// Creates a new zero-filled object of at least `size` bytes with a unique name.
    static SharedMemoryRegion create(size_t size);

    /// Maps the existing object `name` of `size` bytes.
    static SharedMemoryRegion open(const std::string& name, size_t size);

    SharedMemoryRegion() = default;
    ~SharedMemoryRegion();

    SharedMemoryRegion(SharedMemoryRegion&& other) noexcept;
    SharedMemoryRegion& operator=(SharedMemoryRegion&& other) noexcept;
    SharedMemoryRegion(const SharedMemoryRegion&) = delete;
    SharedMemoryRegion& operator=(const SharedMemoryRegion&) = delete;

    /// The shm_open name, at most 31 characters (the macOS limit).
    const std::string& name() const { return regionName; }

    /// The mapped size in bytes (a multiple of kRegionAlignment for created regions).
    size_t size() const { return mappedSize; }

    void* data() { return mapping; }
    const void* data() const { return mapping; }

    template <typename T>
    T* as() {
        return static_cast<T*>(mapping);
    }

private:
    SharedMemoryRegion(std::string name, void* mapping, size_t size, bool owner)
        : regionName(std::move(name)),
          mapping(mapping),
          mappedSize(size),
          owner(owner) {}

    void release();

    /// The shm_open name.
    std::string regionName;
    /// The start of the mapping, or nullptr when empty.
    void* mapping = nullptr;
    /// Bytes mapped.
    size_t mappedSize = 0;
    /// Whether this region created the object and unlinks its name.
    bool owner = false;
};

} // namespace klartraum::ipc

#endif // KLARTRAUM_IPC_SHARED_MEMORY_HPP
