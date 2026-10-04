// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_COSMOS3_TILED_DECODE_HPP
#define KLARTRAUM_COSMOS3_TILED_DECODE_HPP

#include <cstdint>
#include <vector>

namespace klartraum {

/**
 * @brief Start offsets of fixed-size tiles along one axis.
 *
 * Offsets advance by @p stride; the last tile is aligned to the end so that
 * every tile has the full size @p tile and together they cover [0, size).
 * @throws std::runtime_error If tile is zero or larger than size, or stride is zero or larger than tile.
 */
std::vector<uint32_t> tileOffsets(uint32_t size, uint32_t tile, uint32_t stride);

/**
 * @brief Blend weights of one tile along one axis.
 *
 * Linear ramps of @p ramp samples, (i + 0.5) / ramp at the start and
 * (length - i - 0.5) / ramp at the end, capped at one; a side without a
 * neighbouring tile keeps weight one. Two tiles that overlap by exactly
 * @p ramp samples therefore sum to one in the overlap.
 */
std::vector<float> blendRamp(uint32_t length, uint32_t ramp, bool rampStart, bool rampEnd);

/**
 * @brief Accumulates overlapping [C, T, h, w] tiles into a [C, T, H, W] tensor.
 *
 * Each tile is weighted by the product of its row and column ramps, and the
 * result is normalized by the summed weights.
 */
class TileBlender {
public:
    TileBlender(uint32_t channels, uint32_t frames, uint32_t height, uint32_t width, uint32_t ramp);

    /** @brief Adds a tile whose top-left sample is at (@p top, @p left). */
    void add(const std::vector<float>& tile, uint32_t tileHeight, uint32_t tileWidth, uint32_t top, uint32_t left);

    /** @brief The blended tensor. @throws std::runtime_error If a sample was not covered by any tile. */
    std::vector<float> result() const;

private:
    uint32_t channels, frames, height, width, ramp;
    std::vector<float> sum;
    std::vector<float> weight; ///< per (y, x), shared by all channels and frames
};

} // namespace klartraum

#endif // KLARTRAUM_COSMOS3_TILED_DECODE_HPP
