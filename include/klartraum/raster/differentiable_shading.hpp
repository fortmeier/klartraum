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

/**
 * @brief Gradients of the shading driven by per-pixel colour gradients (e.g. from antialiasBackward()).
 *
 * Slots: 7 colour gradients (four floats per pixel, rgb used), 8 vertex gradients, 9 texel gradients (four floats
 * each, added to).
 */
ComputeGraphElementPtr shadeColorBackward(VulkanContext& vulkanContext, uint32_t width, uint32_t height,
                                          uint32_t textureWidth, uint32_t textureHeight);

/**
 * @brief Analytic antialiasing of silhouettes (shaders/raster/antialias.slang): where a mesh edge separates two
 * neighbouring pixels, the pixel partly covered by the other surface blends towards its colour by the covered
 * fraction, which is differentiable in the edge's vertices.
 *
 * Slots: 7 premultiplied colours (four floats per pixel, from shade()), 8 antialiased colours (four floats per
 * pixel).
 */
ComputeGraphElementPtr antialias(VulkanContext& vulkanContext, uint32_t width, uint32_t height, uint32_t textureWidth,
                                 uint32_t textureHeight);

/**
 * @brief Gradients of antialias(): to the colours it blends and, through the silhouette crossings, to the vertices.
 *
 * Slots: 7 premultiplied colours, 8 gradients of the antialiased colours (four floats per pixel), 9 colour gradients
 * (four floats per pixel, added to), 10 vertex gradients (four floats per vertex, added to).
 */
ComputeGraphElementPtr antialiasBackward(VulkanContext& vulkanContext, uint32_t width, uint32_t height,
                                         uint32_t textureWidth, uint32_t textureHeight);

/**
 * @brief The image loss of premultiplied colours against a masked target and its gradients: the sum over pixels of
 * (m |colour - target|^2 + maskWeight (coverage - m)^2) / (width x height), with m the target mask.
 *
 * Slots: 0 colours (four floats per pixel: premultiplied rgb, coverage), 1 targets (four floats per pixel and view:
 * rgb and mask), 2 view index, 3 colour gradients (four floats per pixel, written), 4 loss (one float, added to).
 */
ComputeGraphElementPtr imageLoss(VulkanContext& vulkanContext, uint32_t width, uint32_t height, float maskWeight);

/**
 * @brief imageLoss() with a colour correction per view: the colour c (coverage a) is compared as gain x c +
 * offset x a with the drawn view's gain and offset (rgb), plus viewColorWeight x (|gain - 1|^2 + |offset|^2).
 *
 * Slots: 0-4 as imageLoss(); 5 view colours (eight floats per view: gain rgb, unused, offset rgb, unused), 6 their
 * gradients (eight floats per view, added to; only the drawn view's change).
 */
ComputeGraphElementPtr imageLossWithViewColors(VulkanContext& vulkanContext, uint32_t width, uint32_t height,
                                               float maskWeight, float viewColorWeight);

/**
 * @brief The uniform Laplacian smoothness regularizer of a mesh: weight x sum over vertices of |p_v - mean of the
 * neighbours' positions|^2 / vertexCount, and its gradient.
 *
 * Slots: 0 positions (four floats per vertex), 1 adjacency offsets and 2 neighbours (VertexAdjacency, uint32), 3
 * vertex gradients (four floats per vertex, added to), 4 loss (one float, added to).
 */
ComputeGraphElementPtr laplacianRegularizer(VulkanContext& vulkanContext, uint32_t vertexCount, float weight);

/** @brief Fills every buffer connected to it (slots 0, 1, ...) with zeros. */
ComputeGraphElementPtr clearBuffers(VulkanContext& vulkanContext);

/** @brief Advances a one-element view index on the GPU: view = (view + 1) % viewCount. Slot 0: view index. */
ComputeGraphElementPtr cycleView(VulkanContext& vulkanContext, uint32_t viewCount);

} // namespace klartraum::raster

#endif // KLARTRAUM_RASTER_DIFFERENTIABLE_SHADING_HPP
