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
    uint32_t outputOffset = 0;         ///< channels of the output before the features (e.g. a network's other inputs)
    uint32_t batch = 1;                ///< images, each with its own coordinates, classes and output
    /// Channels of each image's output (0: outputOffset + levels * features), e.g. with more channels after the
    /// features.
    uint32_t outputChannels = 0;

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
 * Slots: 0 coordinates [batch][3][height][width] (u, v, id in [0, 1]), 1 classes [batch][height][width] (class
 * indices as floats), 2 table [grids][levels][tableSize][features], 3 output
 * [batch][outputChannels][height][width] (only channels outputOffset .. + levels * features are written).
 */
ComputeGraphElementPtr hashTexture(VulkanContext& vulkanContext, const HashTextureShape& shape);

/**
 * @brief The table gradient of hashTexture(): every pixel adds its features' gradients, weighted as the lookup blended
 * them, to the 8 corner entries of its cell on each level (atomic float additions, VK_EXT_shader_atomic_float), so
 * the gradient tensor must be zeroed before.
 *
 * Slots: 0 coordinates, 1 classes, 2 output gradient (the output's layout; the features after outputOffset), 3 table
 * gradient (the table's layout).
 */
ComputeGraphElementPtr hashTextureBackward(VulkanContext& vulkanContext, const HashTextureShape& shape);

} // namespace klartraum::neural

#endif // KLARTRAUM_NEURAL_HASH_TEXTURE_HPP
