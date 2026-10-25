// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - sharedMemoryRegionIsAlignedAndZeroed: a created region has a short unique name, a size rounded
 *   up to the region alignment and zeroed contents
 * - sharedMemoryRegionIsSharedBetweenMappings: a second mapping opened by name sees the writes of
 *   the first and the other way round, and the name is gone once the creator is destroyed
 **/
#include <gtest/gtest.h>

#include <cstring>
#include <stdexcept>

#include "klartraum/ipc/shared_memory.hpp"

using namespace klartraum::ipc;

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
