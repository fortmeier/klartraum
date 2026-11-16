// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_RASTER_DIFFERENTIABLE_SHADING_HPP
#define KLARTRAUM_RASTER_DIFFERENTIABLE_SHADING_HPP

#include <cstdint>

#include "klartraum/computegraph/computegraphelement.hpp"
#include "klartraum/vulkan_context.hpp"

/**
 * @brief Differentiable shading of a visibility buffer (see VisibilityPass) and the graph elements a training step
 * needs around it.
 *
 * The shading recomputes, for the triangle a pixel shows, the projection of its vertices, perspective-correct
 * barycentric coordinates at the pixel centre, the interpolated texture coordinate and a bilinear texture lookup.
 * The backward element derives the gradients with Slang's automatic differentiation
 * (shaders/raster/differentiable_shading.slang).
 *
 * The shading elements share their first slots: 0 positions (four floats per vertex), 1 triangles (three uint32
 * each), 2 corner texture coordinates (two floats per triangle corner), 3 texels (four floats each, rgb used,
 * row-major), 4 cameras (column-major 4x4 per view), 5 view index (one uint32), 6 visibility (one uint32 per pixel).
 */
namespace klartraum::raster {

/** @brief Per pixel the textured colour and a coverage of 1, zero for background. Slot 7: colours (four floats per
 * pixel). */
ComputeGraphElementPtr shade(VulkanContext& vulkanContext, uint32_t width, uint32_t height, uint32_t textureWidth,
                             uint32_t textureHeight);

/**
 * @brief The masked colour loss and its gradients: the sum over pixels covered by the mesh and inside the target
 * mask of |colour - target|^2 / (width x height).
 *
 * Slots: 7 targets (four floats per pixel and view: rgb and mask, views one after another), 8 vertex gradients (four
 * floats per vertex), 9 texel gradients (four floats per texel), 10 loss (one float). Gradients and the loss are
 * added to the buffers, which clearBuffers() zeroes before.
 */
ComputeGraphElementPtr shadeBackward(VulkanContext& vulkanContext, uint32_t width, uint32_t height,
                                     uint32_t textureWidth, uint32_t textureHeight);

/** @brief Fills every buffer connected to it (slots 0, 1, ...) with zeros. */
ComputeGraphElementPtr clearBuffers(VulkanContext& vulkanContext);

/** @brief Advances a one-element view index on the GPU: view = (view + 1) % viewCount. Slot 0: view index. */
ComputeGraphElementPtr cycleView(VulkanContext& vulkanContext, uint32_t viewCount);

} // namespace klartraum::raster

#endif // KLARTRAUM_RASTER_DIFFERENTIABLE_SHADING_HPP
