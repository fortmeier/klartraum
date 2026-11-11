// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_GBUFFER_MESH_RENDERER_HPP
#define KLARTRAUM_GBUFFER_MESH_RENDERER_HPP

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include <glm/glm.hpp>

#include "klartraum/computegraph/hostvalues.hpp"
#include "klartraum/computegraph/imageviewsrc.hpp"
#include "klartraum/computegraph/renderpass.hpp"
#include "klartraum/draw_component.hpp"
#include "klartraum/vulkan_context.hpp"

namespace klartraum {

/// A vertex of a GBufferMeshRenderer triangle list; std430 and C++ share the layout.
struct GBufferVertex {
    glm::vec4 pos;    ///< xyz, w the surface class (tree::TREE_CLASS_*)
    glm::vec4 normal; ///< xyz, w the 16-bit object id of the coordinates (0: the surface has none)
    glm::vec4 albedo; ///< rgb, w unused
    glm::vec4 uv;     ///< xy the surface coordinates, z 1 where they repeat (taken modulo 1), w unused
};

/// What a GBufferMeshRenderer draws: the values and encodings of BonsaiOutput.
enum class GBufferOutput : uint32_t {
    Colour = 0,       ///< the albedo, lit by one directional light and an ambient term
    Classes = 3,      ///< the class of every surface as its colour (tree::treeClassColor())
    Depth = 4,        ///< the distance from the camera as grey from near to far (tree::treeGBufferColor())
    CoordinatesU = 6, ///< u as 16 bits (red high byte, green low byte), blue the object id's high byte
    CoordinatesV = 7, ///< v as CoordinatesU, blue the object id's low byte
    Depth16 = 8,      ///< the distance from near to far as 16 bits (red high byte, green low byte)
    Normals = 9,      ///< the world-space normal n as (n + 1) / 2 in red, green and blue
};

/// The values the shaders of a GBufferMeshRenderer read; std430 and C++ share the layout.
struct GBufferMeshSettings {
    glm::vec4 output; ///< x the GBufferOutput, y and z the depth window's near and far distance, w unused
    glm::vec4 light;  ///< xyz the direction towards the light, w the ambient share of the light
};

/**
 * @brief An axis-aligned box from `low` to `high` as 12 triangles with outward normals. Every face has its own
 * coordinates (u, v) in [0, 1], u along the face to the right and v upwards as seen from outside (on the top and
 * bottom faces v runs towards -z and +z), and its own object id: faces +x, -x, +y, -y, +z, -z get `faceIds[0..5]`.
 */
std::vector<GBufferVertex> gbufferBox(glm::vec3 low, glm::vec3 high, uint32_t surfaceClass, glm::vec3 albedo,
                                      const std::array<uint32_t, 6>& faceIds);

/**
 * @brief A square floor at height y, 2 * halfSize wide around the origin (normal +y), whose coordinates repeat every
 * `tile` units in x and z, with one object id.
 */
std::vector<GBufferVertex> gbufferFloor(float halfSize, float y, float tile, uint32_t surfaceClass, glm::vec3 albedo,
                                        uint32_t objectId);

/**
 * @brief Draws a static triangle list into an image: lit, or as one of the G-buffers the bonsai renderer writes
 * (classes, depth, 16-bit depth, surface coordinates with object ids), so that simple scenes (e.g. a toy object on a
 * floor) produce the inputs of the photo enhancer pipeline.
 *
 * `shaders/gbuffer/gbuffer_mesh.{vert,frag}.slang`. The target is cleared to black (class 0, the background) before
 * every draw. Set the camera through getCamera() and update it before submitting the graph of getOutput().
 */
class GBufferMeshRenderer {
public:
    GBufferMeshRenderer(VulkanContext& vulkanContext, const std::vector<GBufferVertex>& vertices,
                        std::shared_ptr<ImageViewSrc> target, VkExtent2D extent);

    /// Selects what the next submits draw; the depth outputs map distances near..far to 0..1.
    void setOutput(GBufferOutput output, float nearDistance = 0.0f, float farDistance = 1.0f);
    /// The light of GBufferOutput::Colour: the direction towards it and the ambient share.
    void setLight(glm::vec3 towardsLight, float ambient);

    std::shared_ptr<CameraUboType> getCamera() const { return camera; }
    /// The render pass; compile a graph from it.
    std::shared_ptr<RenderPass> getOutput() const { return pass; }

private:
    void store();

    std::shared_ptr<CameraUboType> camera;
    /// The output and light the shaders read; set() takes effect with the next submit.
    std::shared_ptr<HostValues<GBufferMeshSettings>> settings;
    GBufferMeshSettings current{};
    std::shared_ptr<RenderPass> pass;
};

} // namespace klartraum

#endif // KLARTRAUM_GBUFFER_MESH_RENDERER_HPP
