// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_NEURAL_HASH_TEXTURE_HPP
#define KLARTRAUM_NEURAL_HASH_TEXTURE_HPP

#include <cstdint>
#include <string>
#include <vector>

#include "klartraum/computegraph/computegraphelement.hpp"
#include "klartraum/vulkan_context.hpp"

namespace klartraum::neural {

/**
 * @brief Shape of a learnable surface texture: one multi-resolution 3D hash grid (Instant NGP) per textured surface
 * class over the surface coordinates (u, v, object id), as scripts/neural/hash_texture.py trains it.
 */
struct HashTextureShape {
    uint32_t width = 0;  ///< image width in pixels
    uint32_t height = 0; ///< image height in pixels
    uint32_t levels = 12;
    uint32_t features = 2;             ///< per level, at most 4
    uint32_t tableSize = 1u << 17;     ///< entries per level, a power of two
    std::vector<uint32_t> resolutions; ///< cells per axis of each level, at most 16 levels
    std::vector<uint32_t> classes;     ///< the surface class of each grid, at most 4 grids

    /**
     * @brief Reads the shape written next to an exported texture (train_photo_enhancer.py: `<stem>_texture.txt`, lines
     * "width W", "height H", "levels L", "features F", "table T", "resolutions R...", "classes C...").
     * @throws std::runtime_error If the file is missing or incomplete.
     */
    static HashTextureShape fromFile(const std::string& path);
};

/**
 * @brief The texture's features of every pixel (`shaders/neural/hash_texture.comp.slang`): from the grid of the
 * pixel's class at its surface coordinates, trilinearly blended from the 8 corners of its cell on each level (direct
 * index where a level's grid fits the table, the spatial hash x * 1 xor y * 2654435761 xor z * 805459861 where it does
 * not), the levels' features side by side; pixels of other classes get zeros.
 *
 * Slots: 0 coordinates [3][height][width] (u, v, id in [0, 1]), 1 classes [height][width] (class indices as floats),
 * 2 table [grids][levels][tableSize][features], 3 output [levels * features][height][width].
 */
ComputeGraphElementPtr hashTexture(VulkanContext& vulkanContext, const HashTextureShape& shape);

} // namespace klartraum::neural

#endif // KLARTRAUM_NEURAL_HASH_TEXTURE_HPP
