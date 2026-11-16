// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_RASTER_MESH_HPP
#define KLARTRAUM_RASTER_MESH_HPP

#include <cstdint>
#include <filesystem>
#include <string>
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

/**
 * @brief The texture of the synthetic pumpkin for its equirectangular corner coordinates, four floats per texel
 * (rgb, 1), row-major: orange with ten darker stripes along the ribs and a green cap around the stem.
 */
std::vector<float> syntheticPumpkinTexture(uint32_t width, uint32_t height);

/**
 * @brief The vertex neighbourhoods of a mesh in compressed sparse rows: the neighbours of vertex v are
 * neighbours[offsets[v]] to neighbours[offsets[v + 1] - 1], each vertex sharing an edge with v once, ascending.
 */
struct VertexAdjacency {
    std::vector<uint32_t> offsets;    ///< vertex count + 1 row starts
    std::vector<uint32_t> neighbours; ///< neighbour vertex indices, row by row
};

/** @brief The vertex adjacency of @p mesh. */
VertexAdjacency vertexAdjacency(const TriangleMesh& mesh);

/**
 * @brief Writes the mesh as a Wavefront OBJ file with corner texture coordinates. With @p textureFile (relative to
 * the OBJ), also writes a material file beside it (same name, .mtl) whose diffuse map is that texture.
 */
void writeObj(const TriangleMesh& mesh, const std::filesystem::path& path, const std::string& textureFile = {});

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
