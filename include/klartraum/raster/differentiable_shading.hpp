// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_RASTER_DIFFERENTIABLE_SHADING_HPP
#define KLARTRAUM_RASTER_DIFFERENTIABLE_SHADING_HPP

#include <cstdint>
#include <vector>

#include <glm/glm.hpp>

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
 * @brief Per pixel the clip w of the nearest triangle's surface at the pixel centre, infinity for background: the
 * stop depth of Gaussians blended in front of the mesh (splatting::SplatForward). Slot 7: depths (one float per
 * pixel).
 */
ComputeGraphElementPtr surfaceDepth(VulkanContext& vulkanContext, uint32_t width, uint32_t height,
                                    uint32_t textureWidth, uint32_t textureHeight);

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

/// Lights of the specular environment (specular()).
constexpr uint32_t kSpecularLights = 4;

/**
 * @brief Adds every triangle's area-weighted normal to its vertices: the sums' directions are the smooth vertex
 * normals (specular()).
 *
 * Slots: 0 positions (four floats per vertex), 1 triangles (three uint32 each), 2 normals (four floats per vertex,
 * added to; clearBuffers() zeroes them before).
 */
ComputeGraphElementPtr vertexNormals(VulkanContext& vulkanContext, uint32_t triangleCount);

/**
 * @brief Adds a specular reflection to the shaded colours of the covered pixels (shaders/raster/specular.slang): per
 * texel a material of a normal offset (a normal map in object space, xyz) and an amplitude a (w); an environment of
 * kSpecularLights lights (direction xyz, amplitude b) and an exponent e = 6 + exp(x) (four floats after them: x, 0, 0,
 * 0),
 * shared by all views. The grey a^2 sum_k b_k^2 max(dot(r, d_k), 0)^e is added, with r the direction to the eye
 * reflected about the smooth normal (the vertex normals interpolated) tilted by the offset.
 *
 * Slots: 0-6 as shade(), 7 colours (four floats per pixel, added to), 8 material (four floats per texel), 9 lights
 * (four floats per light, then four), 10 vertex normals (vertexNormals()), 11 eyes (four floats per view: the
 * camera's position).
 */
ComputeGraphElementPtr specular(VulkanContext& vulkanContext, uint32_t width, uint32_t height, uint32_t textureWidth,
                                uint32_t textureHeight);

/**
 * @brief Gradients of specular() with respect to the material and the lights, from the gradients of the colours.
 *
 * Slots: 0-6 as shade(), 7 colour gradients (four floats per pixel, rgb used), 8-11 as specular(), 12 material
 * gradients (four floats per texel, added to), 13 light gradients (as the lights, added to).
 */
ComputeGraphElementPtr specularBackward(VulkanContext& vulkanContext, uint32_t width, uint32_t height,
                                        uint32_t textureWidth, uint32_t textureHeight);

/** @brief The positions of @p cameras (clip-from-world matrices) as four floats each (xyz, 1), for specular(). */
std::vector<float> cameraEyes(const std::vector<glm::mat4>& cameras);

/**
 * @brief Cameras as poses (shaders/raster/camera_pose.slang): each camera M = P R(q) T(-e), with q a quaternion, e the
 * eye's position and P the fixed rest. Splits @p cameras into their rests (16 floats each, as the cameras are packed)
 * and their poses at rest (eight floats each: q = (0, 0, 0, 1), then e and a zero).
 */
void cameraPoses(const std::vector<glm::mat4>& cameras, std::vector<float>& rests, std::vector<float>& poses);

/**
 * @brief Builds every camera from its pose: M = P R(q) T(-e), and its eye.
 *
 * Slots: 0 rests (16 floats per view), 1 poses (eight floats per view), 2 cameras (16 floats per view, written), 3 eyes
 * (four floats per view, written).
 */
ComputeGraphElementPtr cameraBuild(VulkanContext& vulkanContext, uint32_t viewCount);

/**
 * @brief The gradients of the drawn view's pose from the vertex gradients of the same step (dL/dx with the camera
 * fixed): moving the camera is moving the mesh the other way.
 *
 * Slots: 0 positions (four floats per vertex), 1 vertex gradients (four floats per vertex), 2 poses, 3 view index (one
 * uint32), 4 pose gradients (eight floats per view, added to).
 */
ComputeGraphElementPtr cameraPoseBackward(VulkanContext& vulkanContext, uint32_t vertexCount);

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
 * @brief Premultiplied front colours with coverage over premultiplied back colours: front + (1 - front coverage) x
 * back, e.g. Gaussians blended until a mesh's surface over the antialiased mesh.
 *
 * Slots: 0 front, 1 back, 2 output (four floats per pixel each: rgb, coverage).
 */
ComputeGraphElementPtr compositeOver(VulkanContext& vulkanContext, uint32_t width, uint32_t height);

/**
 * @brief Gradients of compositeOver() with respect to its front and back colours from the gradients of its output.
 *
 * Slots: 0 front, 1 back, 2 output gradients, 3 front gradients (written), 4 back gradients (written); four floats
 * per pixel each.
 */
ComputeGraphElementPtr compositeOverBackward(VulkanContext& vulkanContext, uint32_t width, uint32_t height);

/**
 * @brief The image loss of premultiplied colours against a masked target and its gradients: the sum over pixels of
 * (m |colour - target|^2 + maskWeight (coverage - m)^2) / (width x height), with m the target mask.
 *
 * With @p highlightTolerance t > 0, the colour term of a pixel whose target is brighter (in luminance) than its colour
 * by more than t is weighted by (t / excess)^2, a weight the gradient takes as constant: highlights that a single view
 * sees (specular reflections) pull a colour much less than what the views agree on.
 *
 * Slots: 0 colours (four floats per pixel: premultiplied rgb, coverage), 1 targets (four floats per pixel and view:
 * rgb and mask), 2 view index, 3 colour gradients (four floats per pixel, written), 4 loss (one float, added to).
 */
ComputeGraphElementPtr imageLoss(VulkanContext& vulkanContext, uint32_t width, uint32_t height, float maskWeight,
                                 float highlightTolerance = 0.0f);

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
