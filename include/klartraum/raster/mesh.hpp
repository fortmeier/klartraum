// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_RASTER_MESH_HPP
#define KLARTRAUM_RASTER_MESH_HPP

#include <cstdint>
#include <filesystem>
#include <vector>

#include <glm/glm.hpp>

namespace klartraum::raster {

/**
 * @brief A triangle mesh with shared vertex positions and texture coordinates per triangle corner.
 *
 * Shared positions keep the surface closed while an optimizer moves them; corner texture coordinates let a
 * triangle cross the seam of an equirectangular texture (u wraps around).
 */
struct TriangleMesh {
    std::vector<glm::vec3> positions;
    std::vector<glm::uvec3> triangles; ///< vertex indices, counter-clockwise seen from outside
    std::vector<glm::vec2> cornerUvs;  ///< three per triangle, u in [0, 2) for seam-crossing triangles, v in [0, 1]
};

/**
 * @brief A unit icosphere: an icosahedron with every triangle split into four @p subdivisions times, projected
 * onto the sphere. Corner texture coordinates are equirectangular: u = longitude / 2 pi, v = polar angle / pi.
 */
TriangleMesh icosphere(uint32_t subdivisions);

/**
 * @brief The synthetic pumpkin of the tests and the demo: the unit icosphere scaled per vertex by
 * 1 + ribDepth * cos(ribs * longitude) * sin(polar angle), flattened to @p height along y, with a stem bump at the
 * top.
 */
TriangleMesh syntheticPumpkin(uint32_t subdivisions, uint32_t ribs = 10, float ribDepth = 0.08f, float height = 0.75f);

/** @brief Writes the mesh as a Wavefront OBJ file with corner texture coordinates. */
void writeObj(const TriangleMesh& mesh, const std::filesystem::path& path);

/** @brief A Vulkan clip-from-view perspective matrix (right-handed, camera looks down -z, y down, depth 0 to 1). */
glm::mat4 perspective(float verticalFov, float aspect, float nearPlane, float farPlane);

/**
 * @brief Clip-from-world matrices of @p count cameras evenly spaced on a horizontal ring of radius @p distance at
 * elevation @p elevation (radians, positive above the equator), looking at the origin.
 */
std::vector<glm::mat4> cameraRing(uint32_t count, float distance, float elevation, float verticalFov, float aspect,
                                  float startAngle = 0.0f);

} // namespace klartraum::raster

#endif // KLARTRAUM_RASTER_MESH_HPP
