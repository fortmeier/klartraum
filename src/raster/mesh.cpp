// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/raster/mesh.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <stdexcept>
#include <utility>

#include <glm/gtc/matrix_transform.hpp>

namespace klartraum::raster {

namespace {

constexpr float kPi = 3.14159265358979f;

float longitude(const glm::vec3& p) { return std::atan2(p.z, p.x); }
float polar(const glm::vec3& p) { return std::acos(std::clamp(p.y / glm::length(p), -1.0f, 1.0f)); }

} // namespace

TriangleMesh icosphere(uint32_t subdivisions) {
    const float t = (1.0f + std::sqrt(5.0f)) / 2.0f;
    std::vector<glm::vec3> positions = {{-1, t, 0},  {1, t, 0},  {-1, -t, 0}, {1, -t, 0}, {0, -1, t},  {0, 1, t},
                                        {0, -1, -t}, {0, 1, -t}, {t, 0, -1},  {t, 0, 1},  {-t, 0, -1}, {-t, 0, 1}};
    std::vector<glm::uvec3> triangles = {{0, 11, 5}, {0, 5, 1},  {0, 1, 7},   {0, 7, 10}, {0, 10, 11},
                                         {1, 5, 9},  {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
                                         {3, 9, 4},  {3, 4, 2},  {3, 2, 6},   {3, 6, 8},  {3, 8, 9},
                                         {4, 9, 5},  {2, 4, 11}, {6, 2, 10},  {8, 6, 7},  {9, 8, 1}};
    for (auto& p : positions)
        p = glm::normalize(p);
    for (uint32_t level = 0; level < subdivisions; ++level) {
        std::map<std::pair<uint32_t, uint32_t>, uint32_t> midpoints;
        auto midpoint = [&](uint32_t a, uint32_t b) {
            const auto key = std::minmax(a, b);
            const auto found = midpoints.find(key);
            if (found != midpoints.end())
                return found->second;
            positions.push_back(glm::normalize(positions[a] + positions[b]));
            return midpoints[key] = uint32_t(positions.size() - 1);
        };
        std::vector<glm::uvec3> finer;
        finer.reserve(triangles.size() * 4);
        for (const auto& tri : triangles) {
            const uint32_t ab = midpoint(tri.x, tri.y), bc = midpoint(tri.y, tri.z), ca = midpoint(tri.z, tri.x);
            finer.push_back({tri.x, ab, ca});
            finer.push_back({tri.y, bc, ab});
            finer.push_back({tri.z, ca, bc});
            finer.push_back({ab, bc, ca});
        }
        triangles = std::move(finer);
    }

    // Equirectangular corner coordinates. A triangle spanning the seam (longitude +-pi) moves its corners on the far
    // side by one turn, so u grows continuously across it; a pole corner takes the longitude of the triangle's other
    // corners.
    TriangleMesh mesh;
    mesh.positions = positions;
    mesh.triangles = triangles;
    mesh.cornerUvs.reserve(triangles.size() * 3);
    for (const auto& tri : triangles) {
        const glm::vec3 p[3] = {positions[tri.x], positions[tri.y], positions[tri.z]};
        float u[3], v[3];
        for (int c = 0; c < 3; ++c) {
            u[c] = (longitude(p[c]) + kPi) / (2 * kPi);
            v[c] = polar(p[c]) / kPi;
        }
        const float maxU = std::max({u[0], u[1], u[2]});
        for (int c = 0; c < 3; ++c)
            if (maxU - u[c] > 0.5f)
                u[c] += 1.0f;
        for (int c = 0; c < 3; ++c) {
            if (std::abs(p[c].y) > 0.9999f) { // a pole: the mean longitude of the other two corners
                float sum = 0.0f;
                for (int o = 0; o < 3; ++o)
                    if (o != c)
                        sum += u[o];
                u[c] = sum / 2.0f;
            }
            mesh.cornerUvs.push_back({u[c], v[c]});
        }
    }
    return mesh;
}

TriangleMesh syntheticPumpkin(uint32_t subdivisions, uint32_t ribs, float ribDepth, float height) {
    TriangleMesh mesh = icosphere(subdivisions);
    for (auto& p : mesh.positions) {
        const float theta = polar(p), phi = longitude(p);
        const float rib = 1.0f + ribDepth * std::cos(float(ribs) * phi) * std::sin(theta);
        const float stem = 0.25f * std::exp(-theta * theta / 0.02f); // a bump within about 8 degrees of the top
        p = glm::vec3(p.x * rib, p.y * height * rib + stem, p.z * rib);
    }
    return mesh;
}

VertexAdjacency vertexAdjacency(const TriangleMesh& mesh) {
    std::vector<std::vector<uint32_t>> rows(mesh.positions.size());
    for (const auto& t : mesh.triangles)
        for (int c = 0; c < 3; ++c) {
            rows[t[c]].push_back(t[(c + 1) % 3]);
            rows[t[(c + 1) % 3]].push_back(t[c]);
        }
    VertexAdjacency adjacency;
    adjacency.offsets.push_back(0);
    for (auto& row : rows) {
        std::sort(row.begin(), row.end());
        row.erase(std::unique(row.begin(), row.end()), row.end());
        adjacency.neighbours.insert(adjacency.neighbours.end(), row.begin(), row.end());
        adjacency.offsets.push_back(uint32_t(adjacency.neighbours.size()));
    }
    return adjacency;
}

std::vector<float> syntheticPumpkinTexture(uint32_t width, uint32_t height) {
    std::vector<float> texels;
    texels.reserve(4 * size_t(width) * height);
    for (uint32_t j = 0; j < height; ++j)
        for (uint32_t i = 0; i < width; ++i) {
            const float u = (float(i) + 0.5f) / float(width), v = (float(j) + 0.5f) / float(height);
            const float stripe = 0.75f + 0.25f * std::cos(10.0f * 2.0f * glm::pi<float>() * u);
            const glm::vec3 color = v < 0.12f ? glm::vec3(0.25f, 0.45f, 0.15f) : stripe * glm::vec3(0.95f, 0.5f, 0.1f);
            texels.insert(texels.end(), {color.r, color.g, color.b, 1.0f});
        }
    return texels;
}

void writeObj(const TriangleMesh& mesh, const std::filesystem::path& path, const std::string& textureFile) {
    std::ofstream out(path);
    if (!out)
        throw std::runtime_error("writeObj: cannot open " + path.string());
    if (!textureFile.empty()) {
        std::filesystem::path materialPath = path;
        materialPath.replace_extension(".mtl");
        std::ofstream material(materialPath);
        if (!material)
            throw std::runtime_error("writeObj: cannot open " + materialPath.string());
        material << "newmtl surface\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nmap_Kd " << textureFile << "\n";
        out << "mtllib " << materialPath.filename().string() << "\nusemtl surface\n";
    }
    for (const auto& p : mesh.positions)
        out << "v " << p.x << " " << p.y << " " << p.z << "\n";
    for (const auto& uv : mesh.cornerUvs)
        out << "vt " << uv.x << " " << 1.0f - uv.y << "\n";
    for (size_t t = 0; t < mesh.triangles.size(); ++t) {
        const auto& tri = mesh.triangles[t];
        out << "f " << tri.x + 1 << "/" << 3 * t + 1 << " " << tri.y + 1 << "/" << 3 * t + 2 << " " << tri.z + 1 << "/"
            << 3 * t + 3 << "\n";
    }
}

glm::mat4 perspective(float verticalFov, float aspect, float nearPlane, float farPlane) {
    glm::mat4 projection = glm::perspectiveRH_ZO(verticalFov, aspect, nearPlane, farPlane);
    projection[1][1] *= -1.0f; // Vulkan's y points down
    return projection;
}

std::vector<glm::mat4> cameraRing(uint32_t count, float distance, float elevation, float verticalFov, float aspect,
                                  float startAngle) {
    std::vector<glm::mat4> cameras;
    const glm::mat4 projection = perspective(verticalFov, aspect, 0.1f, 10.0f * distance);
    for (uint32_t i = 0; i < count; ++i) {
        const float angle = startAngle + 2.0f * kPi * float(i) / float(count);
        const glm::vec3 eye = distance * glm::vec3(std::cos(elevation) * std::cos(angle), std::sin(elevation),
                                                   std::cos(elevation) * std::sin(angle));
        cameras.push_back(projection * glm::lookAt(eye, glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f)));
    }
    return cameras;
}

} // namespace klartraum::raster
