// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - Tile offsets advance by the stride, end-align the last tile, and reduce to one tile when it fits.
 * - Invalid tile and stride sizes are rejected.
 * - Blend ramps rise and fall linearly, and keep weight one on sides without a neighbour.
 * - Ramps of tiles that overlap by the ramp length sum to one in the overlap.
 * - A single tile covering the tensor is returned unchanged.
 * - Tiles cut from a known [C, T, H, W] tensor blend back to it exactly (3x3 tiles, Cosmos3 layout).
 * - Uncovered samples are rejected.
 **/

#include <cstdint>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "klartraum/cosmos3/tiled_decode.hpp"

using klartraum::blendRamp;
using klartraum::TileBlender;
using klartraum::tileOffsets;

namespace {

/** Copies the [C, T, h, w] window at (top, left) out of a [C, T, H, W] tensor. */
std::vector<float> cutTile(const std::vector<float>& tensor, uint32_t channels, uint32_t frames, uint32_t height,
                           uint32_t width, uint32_t tileHeight, uint32_t tileWidth, uint32_t top, uint32_t left) {
    std::vector<float> tile;
    tile.reserve(size_t(channels) * frames * tileHeight * tileWidth);
    for (uint32_t ct = 0; ct < channels * frames; ++ct) {
        for (uint32_t y = 0; y < tileHeight; ++y) {
            for (uint32_t x = 0; x < tileWidth; ++x) {
                tile.push_back(tensor[(size_t(ct) * height + top + y) * width + left + x]);
            }
        }
    }
    return tile;
}

} // namespace

TEST(Cosmos3TiledDecodeTest, tileOffsetsCoverTheAxis) {
    EXPECT_EQ(tileOffsets(32, 16, 8), (std::vector<uint32_t>{0, 8, 16}));
    EXPECT_EQ(tileOffsets(30, 16, 8), (std::vector<uint32_t>{0, 8, 14}));
    EXPECT_EQ(tileOffsets(32, 16, 16), (std::vector<uint32_t>{0, 16}));
    EXPECT_EQ(tileOffsets(16, 16, 8), (std::vector<uint32_t>{0}));
}

TEST(Cosmos3TiledDecodeTest, invalidTilingIsRejected) {
    EXPECT_THROW(tileOffsets(8, 16, 8), std::runtime_error);
    EXPECT_THROW(tileOffsets(32, 0, 8), std::runtime_error);
    EXPECT_THROW(tileOffsets(32, 16, 0), std::runtime_error);
    EXPECT_THROW(tileOffsets(32, 16, 17), std::runtime_error);
}

TEST(Cosmos3TiledDecodeTest, blendRampShape) {
    EXPECT_EQ(blendRamp(4, 2, false, false), (std::vector<float>{1.0f, 1.0f, 1.0f, 1.0f}));
    EXPECT_EQ(blendRamp(4, 2, true, false), (std::vector<float>{0.25f, 0.75f, 1.0f, 1.0f}));
    EXPECT_EQ(blendRamp(4, 2, false, true), (std::vector<float>{1.0f, 1.0f, 0.75f, 0.25f}));
    EXPECT_EQ(blendRamp(4, 2, true, true), (std::vector<float>{0.25f, 0.75f, 0.75f, 0.25f}));
}

TEST(Cosmos3TiledDecodeTest, overlappingRampsSumToOne) {
    // Tiles of 256 samples at 0 and 128 overlap by the ramp length 128.
    const auto first = blendRamp(256, 128, false, true);
    const auto second = blendRamp(256, 128, true, true);
    for (uint32_t i = 0; i < 128; ++i)
        EXPECT_FLOAT_EQ(first[128 + i] + second[i], 1.0f) << "sample " << i;
}

TEST(Cosmos3TiledDecodeTest, singleTileIsUnchanged) {
    std::vector<float> tensor(2 * 3 * 4 * 5);
    for (size_t i = 0; i < tensor.size(); ++i)
        tensor[i] = float(i) * 0.25f - 3.0f;
    TileBlender blender(2, 3, 4, 5, 2);
    blender.add(tensor, 4, 5, 0, 0);
    EXPECT_EQ(blender.result(), tensor);
}

TEST(Cosmos3TiledDecodeTest, tilesOfAKnownTensorBlendBackExactly) {
    // Cosmos3 512 layout, scaled down: 32x32 samples, 16x16 tiles at stride 8, ramp = overlap = 8.
    const uint32_t channels = 3, frames = 2, size = 32, tile = 16, stride = 8;
    std::vector<float> tensor(size_t(channels) * frames * size * size);
    for (size_t i = 0; i < tensor.size(); ++i)
        tensor[i] = float((i * 7919) % 1000) / 500.0f - 1.0f;
    TileBlender blender(channels, frames, size, size, tile - stride);
    const auto offsets = tileOffsets(size, tile, stride);
    ASSERT_EQ(offsets.size(), 3u);
    for (uint32_t top : offsets) {
        for (uint32_t left : offsets) {
            blender.add(cutTile(tensor, channels, frames, size, size, tile, tile, top, left), tile, tile, top, left);
        }
    }
    const auto blended = blender.result();
    ASSERT_EQ(blended.size(), tensor.size());
    for (size_t i = 0; i < tensor.size(); ++i)
        EXPECT_NEAR(blended[i], tensor[i], 1e-6f) << "element " << i;
}

TEST(Cosmos3TiledDecodeTest, uncoveredSamplesAreRejected) {
    TileBlender blender(1, 1, 4, 4, 2);
    blender.add(std::vector<float>(4, 1.0f), 2, 2, 0, 0);
    EXPECT_THROW(blender.result(), std::runtime_error);
}
