// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/cosmos3/tiled_decode.hpp"

#include <algorithm>
#include <stdexcept>

namespace klartraum {

std::vector<uint32_t> tileOffsets(uint32_t size, uint32_t tile, uint32_t stride) {
    if (tile == 0 || tile > size || stride == 0 || stride > tile) {
        throw std::runtime_error("tileOffsets: invalid tiling");
    }
    std::vector<uint32_t> offsets;
    for (uint32_t offset = 0; offset + tile < size; offset += stride)
        offsets.push_back(offset);
    offsets.push_back(size - tile);
    return offsets;
}

std::vector<float> blendRamp(uint32_t length, uint32_t ramp, bool rampStart, bool rampEnd) {
    std::vector<float> weights(length, 1.0f);
    for (uint32_t i = 0; i < length; ++i) {
        if (rampStart)
            weights[i] = std::min(weights[i], (float(i) + 0.5f) / float(ramp));
        if (rampEnd)
            weights[i] = std::min(weights[i], (float(length - i) - 0.5f) / float(ramp));
    }
    return weights;
}

TileBlender::TileBlender(uint32_t channels, uint32_t frames, uint32_t height, uint32_t width, uint32_t ramp)
    : channels(channels),
      frames(frames),
      height(height),
      width(width),
      ramp(ramp),
      sum(size_t(channels) * frames * height * width, 0.0f),
      weight(size_t(height) * width, 0.0f) {}

void TileBlender::add(const std::vector<float>& tile, uint32_t tileHeight, uint32_t tileWidth, uint32_t top,
                      uint32_t left) {
    if (top + tileHeight > height || left + tileWidth > width ||
        tile.size() != size_t(channels) * frames * tileHeight * tileWidth) {
        throw std::runtime_error("TileBlender: tile does not fit");
    }
    // Ramps only towards neighbouring tiles; the tensor borders keep weight one.
    const auto rows = blendRamp(tileHeight, ramp, top > 0, top + tileHeight < height);
    const auto columns = blendRamp(tileWidth, ramp, left > 0, left + tileWidth < width);
    for (uint32_t y = 0; y < tileHeight; ++y) {
        for (uint32_t x = 0; x < tileWidth; ++x) {
            weight[size_t(top + y) * width + left + x] += rows[y] * columns[x];
        }
    }
    for (uint32_t ct = 0; ct < channels * frames; ++ct) {
        for (uint32_t y = 0; y < tileHeight; ++y) {
            const float* source = tile.data() + (size_t(ct) * tileHeight + y) * tileWidth;
            float* target = sum.data() + (size_t(ct) * height + top + y) * width + left;
            for (uint32_t x = 0; x < tileWidth; ++x)
                target[x] += rows[y] * columns[x] * source[x];
        }
    }
}

std::vector<float> TileBlender::result() const {
    if (std::any_of(weight.begin(), weight.end(), [](float w) { return w <= 0.0f; })) {
        throw std::runtime_error("TileBlender: not every sample is covered by a tile");
    }
    std::vector<float> blended(sum.size());
    const size_t plane = size_t(height) * width;
    for (size_t i = 0; i < blended.size(); ++i)
        blended[i] = sum[i] / weight[i % plane];
    return blended;
}

} // namespace klartraum
