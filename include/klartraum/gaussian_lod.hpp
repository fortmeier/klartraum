// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_GAUSSIAN_LOD_HPP
#define KLARTRAUM_GAUSSIAN_LOD_HPP

#include <array>
#include <cstdint>
#include <vector>

#include "klartraum/rendering/vulkan_gaussian_splatting_types.hpp"

namespace klartraum {

/**
 * @brief Reduces Gaussians to at most `budget` while keeping where they are and how much they cover.
 *
 * After the level of detail of klartraum-spacegame (tools/asset_lab,
 * spatial_reduce()): the finest grid whose occupied cells fit the budget is
 * found, every cell keeps its most visible Gaussian (optical depth
 * -log(1 - alpha) times the area of its two broad axes), and the rest of the
 * budget goes evenly to the others. The kept Gaussians are source Gaussians
 * with their colours and orientations; only their two broad axes grow, by the
 * square root of their cell's lost coverage (at most 3 times), so the cell
 * covers as much as before. The thin axis stays.
 */
std::vector<Gaussian3D> gaussianSpatialReduce(const std::vector<Gaussian3D>& gaussians, uint32_t budget);

/// Four levels of detail of a set of Gaussians, from the full set (level 0) down.
struct GaussianLods {
    std::array<std::vector<Gaussian3D>, 4> levels;
};

/**
 * @brief Levels 1 to 3 as gaussianSpatialReduce() of the level before, with 1/4, 1/16 and 1/64 of the full count.
 *
 * At least `minimum` Gaussians are kept on every level. All levels share
 * the full set's coordinates, so switching them does not move anything.
 */
GaussianLods buildGaussianLods(const std::vector<Gaussian3D>& full, uint32_t minimum = 16);

} // namespace klartraum

#endif // KLARTRAUM_GAUSSIAN_LOD_HPP
