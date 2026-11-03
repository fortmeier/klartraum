// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/tree/bonsai_scene.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "klartraum/raster/mesh.hpp"
#include "klartraum/rendering/gaussian_data_standard.hpp"
#include "klartraum/tree/tree_placement.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <functional>
#include <limits>
#include <stdexcept>

namespace klartraum {

namespace {

constexpr float kPi = 3.14159265f;

float signedPow(float value, float exponent) { return std::copysign(std::pow(std::abs(value), exponent), value); }

// The pot's outline at angle theta, scaled by `scale`.
glm::vec2 potOutline(const BonsaiPot& pot, float theta, float scale) {
    const float e = 2.0f / pot.exponent;
    return scale *
           glm::vec2(pot.halfLength * signedPow(std::cos(theta), e), pot.halfWidth * signedPow(std::sin(theta), e));
}

// The scale of the inner outline, where the wall ends at the rim.
float innerScale(const BonsaiPot& pot) { return 1.0f - pot.wall / pot.halfWidth; }

using SurfaceFn = std::function<glm::vec3(float u, float v)>;

// A grid surface over [0, 1]^2 as triangles, with normals from the surface
// derivatives, turned to face `outside(p)`.
void addSurface(std::vector<TreeSceneVertex>& mesh, const SurfaceFn& surface, int nu, int nv, glm::vec4 color,
                const std::function<glm::vec3(glm::vec3)>& outside) {
    auto normalAt = [&](float u, float v) {
        const float e = 1e-3f;
        const glm::vec3 du = surface(std::min(u + e, 1.0f), v) - surface(std::max(u - e, 0.0f), v);
        const glm::vec3 dv = surface(u, std::min(v + e, 1.0f)) - surface(u, std::max(v - e, 0.0f));
        glm::vec3 n = glm::cross(du, dv);
        const float l = glm::length(n);
        n = l > 1e-12f ? n / l : glm::vec3(0.0f, 1.0f, 0.0f);
        const glm::vec3 p = surface(u, v);
        return glm::dot(n, outside(p)) < 0.0f ? -n : n;
    };
    auto vertex = [&](float u, float v) {
        TreeSceneVertex vertex{};
        vertex.pos = glm::vec4(surface(u, v), 1.0f);
        vertex.normal = glm::vec4(normalAt(u, v), 0.0f);
        vertex.color = color;
        return vertex;
    };
    for (int j = 0; j < nv; ++j) {
        for (int i = 0; i < nu; ++i) {
            const float u0 = float(i) / nu, u1 = float(i + 1) / nu;
            const float v0 = float(j) / nv, v1 = float(j + 1) / nv;
            const TreeSceneVertex a = vertex(u0, v0), b = vertex(u1, v0), c = vertex(u1, v1), d = vertex(u0, v1);
            mesh.insert(mesh.end(), {a, b, c, a, c, d});
        }
    }
}

float hashUnit(uint32_t seed, uint32_t key) { return tree::treeRand(seed, key); }

// The value below which `fraction` of `values` lie.
float quantile(std::vector<float> values, float fraction) {
    const size_t k = std::min(values.size() - 1, size_t(fraction * float(values.size() - 1) + 0.5f));
    std::nth_element(values.begin(), values.begin() + k, values.end());
    return values[k];
}

// The smallest rectangle (robustly: 1 % to 99 % of the points along each side) around `points` in the plane:
// x its half length, y its half width, z the angle about +y of its long axis from +x; and its middle.
std::pair<glm::vec3, glm::vec2> smallestRectangle(const std::vector<glm::vec2>& points) {
    float bestArea = std::numeric_limits<float>::max();
    std::pair<glm::vec3, glm::vec2> best;
    for (int step = 0; step < 90; ++step) {
        const float angle = float(step) * kPi / 180.0f;
        const glm::vec2 along(std::cos(angle), std::sin(angle)), across(-along.y, along.x);
        std::vector<float> a, b;
        for (const auto& p : points) {
            a.push_back(glm::dot(p, along));
            b.push_back(glm::dot(p, across));
        }
        const float a0 = quantile(a, 0.01f), a1 = quantile(a, 0.99f), b0 = quantile(b, 0.01f), b1 = quantile(b, 0.99f);
        if ((a1 - a0) * (b1 - b0) >= bestArea)
            continue;
        bestArea = (a1 - a0) * (b1 - b0);
        const glm::vec2 middle = 0.5f * (a0 + a1) * along + 0.5f * (b0 + b1) * across;
        // The long axis (x, z) = along or across; turning +x by `yaw` about +y gives (cos yaw, -sin yaw).
        const bool alongIsLong = a1 - a0 >= b1 - b0;
        const glm::vec2 axis = alongIsLong ? along : across;
        best = {glm::vec3(0.5f * std::max(a1 - a0, b1 - b0), 0.5f * std::min(a1 - a0, b1 - b0),
                          std::atan2(-axis.y, axis.x)),
                middle};
    }
    return best;
}

// Bounds (0.5 % to 99.5 % per axis, so stray Gaussians and spikes hardly count) and the rim.
void measureObject(BonsaiObject& object, const std::vector<glm::vec3>& points) {
    if (points.empty())
        throw std::runtime_error("loadBonsaiObject: the object is empty");
    for (int axis = 0; axis < 3; ++axis) {
        std::vector<float> values;
        for (const auto& p : points)
            values.push_back(p[axis]);
        object.low[axis] = quantile(values, 0.005f);
        object.high[axis] = quantile(values, 0.995f);
    }
    // Horizontal cross-sections: the area of the smallest rectangle around each slice.
    const int slices = 40;
    const float sliceHeight = (object.high.y - object.low.y) / float(slices);
    std::vector<std::vector<glm::vec2>> slicePoints(slices);
    for (const auto& p : points) {
        const int k = int((p.y - object.low.y) / sliceHeight);
        if (k >= 0 && k < slices)
            slicePoints[k].emplace_back(p.x, p.z);
    }
    std::vector<float> area(slices, 0.0f);
    float widest = 0.0f;
    for (int k = 0; k < slices; ++k) {
        if (slicePoints[k].size() < 20)
            continue;
        const glm::vec3 r = smallestRectangle(slicePoints[k]).first;
        area[k] = 4.0f * r.x * r.y;
        if (k >= slices / 2)
            widest = std::max(widest, area[k]);
    }
    const float widestOfAll = *std::max_element(area.begin(), area.end());
    int footSlice = 0;
    while (footSlice < slices - 1 && area[footSlice] < 0.5f * widestOfAll)
        ++footSlice;
    object.footHeight = object.low.y + float(footSlice) * sliceHeight;
    int rimSlice = slices - 1;
    while (rimSlice > slices / 2 && area[rimSlice] < 0.9f * widest)
        --rimSlice;
    object.rimHeight = object.low.y + float(rimSlice + 1) * sliceHeight;
    // The rim: the slices of its top 8 % of the height.
    std::vector<glm::vec2> rimPoints;
    for (int k = std::max(0, rimSlice - 2); k <= rimSlice; ++k)
        rimPoints.insert(rimPoints.end(), slicePoints[k].begin(), slicePoints[k].end());
    const auto [rectangle, middle] = smallestRectangle(rimPoints);
    object.rim = rectangle;
    object.rimCenter = middle;
}

// Appends `object` to the scene, mapped by `transform` (rotation about +y, uniform `scale`, translation).
void placeObject(BonsaiScene& scene, const BonsaiObject& object, const glm::mat4& transform, float scale,
                 const std::function<bool(const TreeSceneVertex*)>& keepTriangle = nullptr,
                 const std::function<bool(const Gaussian3D&)>& keepGaussian = nullptr) {
    const glm::quat turn = glm::quat_cast(glm::mat3(transform) / scale);
    for (Gaussian3D g : object.gaussians) {
        if (keepGaussian && !keepGaussian(g))
            continue;
        const glm::vec3 p = glm::vec3(transform * glm::vec4(g.position[0], g.position[1], g.position[2], 1.0f));
        const glm::quat q = turn * glm::quat(g.rotation[3], g.rotation[0], g.rotation[1], g.rotation[2]);
        g.position = {p.x, p.y, p.z};
        g.rotation = {q.x, q.y, q.z, q.w};
        for (float& axis : g.scale)
            axis *= scale;
        scene.objectGaussians.push_back(g);
    }
    for (size_t t = 0; t + 2 < object.mesh.size(); t += 3) {
        if (keepTriangle && !keepTriangle(&object.mesh[t]))
            continue;
        for (size_t c = 0; c < 3; ++c) {
            TreeSceneVertex v = object.mesh[t + c];
            v.pos = glm::vec4(glm::vec3(transform * glm::vec4(glm::vec3(v.pos), 1.0f)), 0.0f);
            v.normal = glm::vec4(turn * glm::vec3(v.normal), 0.0f);
            scene.mesh.push_back(v);
        }
    }
}

// Maps an object so that its bottom centre `anchor` stands at `at`, turned by `yaw` about +y and scaled by `scale`.
glm::mat4 objectTransform(glm::vec3 anchor, glm::vec3 at, float yaw, float scale) {
    return glm::translate(glm::mat4(1.0f), at) * glm::rotate(glm::mat4(1.0f), yaw, glm::vec3(0.0f, 1.0f, 0.0f)) *
           glm::scale(glm::mat4(1.0f), glm::vec3(scale)) * glm::translate(glm::mat4(1.0f), -anchor);
}

} // namespace

float BonsaiScale::regionMeters() const { return kSmallestRegion * std::pow(kLargestRegion / kSmallestRegion, s); }

float BonsaiScale::lanternMeters() const {
    const float t = std::clamp(s / kForest, 0.0f, 1.0f);
    return 0.35f * std::pow(1.8f / 0.35f, t);
}

float BonsaiScale::treeHeightUnits() const {
    const float atForest = 10.7f / BonsaiScale{kForest}.metersPerUnit();
    if (s <= kForest) {
        // From the bonsai filling its pot to the young trees of a planting.
        return 2.0f + (atForest - 2.0f) * std::max(s, 0.0f) / kForest;
    }
    const float t = std::clamp((s - kForest) / (0.4f - kForest), 0.0f, 1.0f);
    return (10.7f + (25.0f - 10.7f) * t * t * (3.0f - 2.0f * t)) / metersPerUnit();
}

float BonsaiScale::bonsaiStyle() const {
    const float t = std::clamp(s / kForest, 0.0f, 1.0f);
    return 1.0f - t * t * (3.0f - 2.0f * t);
}

uint32_t BonsaiScale::treeCount(float soilArea) const {
    const float squareMeters = soilArea * metersPerUnit() * metersPerUnit();
    return std::max(1u, uint32_t(std::lround(kTreesPerHectare * squareMeters / 1e4f)));
}

float BonsaiScale::reliefUnits() const {
    const float t = std::clamp((s - kReliefStart) / (1.0f - kReliefStart), 0.0f, 1.0f);
    const float atLargest = kReliefMeters / BonsaiScale{1.0f}.metersPerUnit();
    return reliefExaggeration * atLargest * t * t * (3.0f - 2.0f * t);
}

float BonsaiScale::reliefRidges() const {
    const float t = std::clamp((s - 0.4f) / 0.6f, 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

float BonsaiScale::forestShare() const {
    const float t = std::clamp((s - 0.3f) / 0.3f, 0.0f, 1.0f);
    return 1.0f - 0.5f * t * t * (3.0f - 2.0f * t);
}

float calibrateForestThreshold(const TreeTerrainParams& terrain, float share) {
    if (share >= 1.0f) {
        return -10.0f;
    }
    // What decides the land cover, once per grid point; only the threshold changes during the bisection.
    std::vector<glm::vec4> sites;
    const int nx = 144, nz = 88;
    for (int j = 0; j < nz; ++j) {
        for (int i = 0; i < nx; ++i) {
            const float x = terrain.outline.x * (2.0f * (i + 0.5f) / nx - 1.0f);
            const float z = terrain.outline.y * (2.0f * (j + 0.5f) / nz - 1.0f);
            if (tree::treeTerrainRadius(terrain, x, z) < 1.0f) {
                sites.push_back(tree::treeLandSite(terrain, x, z));
            }
        }
    }
    TreeTerrainParams t = terrain;
    auto forestShare = [&](float threshold) {
        t.cover.x = threshold;
        double sum = 0.0;
        for (const glm::vec4& site : sites) {
            sum += tree::treeLandCoverFrom(t, site).vegetation.x;
        }
        return float(sum / double(sites.size()));
    };
    // The forest shrinks as the threshold rises.
    float low = -1.0f, high = 2.5f;
    for (int i = 0; i < 40; ++i) {
        const float middle = 0.5f * (low + high);
        (forestShare(middle) > share ? low : high) = middle;
    }
    return 0.5f * (low + high);
}

float BonsaiScene::potMeasure(float x, float z, float scale) const {
    const BonsaiPot& pot = config.pot;
    return std::pow(std::abs(x / (pot.halfLength * scale)), pot.exponent) +
           std::pow(std::abs(z / (pot.halfWidth * scale)), pot.exponent);
}

float BonsaiScene::soilHeight(float x, float z) const { return tree::treeTerrainHeight(terrain, x, z); }

BonsaiObject bonsaiObjectFromMesh(const raster::TriangleMesh& mesh, const std::vector<float>& texels, uint32_t width,
                                  uint32_t height) {
    BonsaiObject object;
    // Smooth normals: the area-weighted face normals around each vertex.
    std::vector<glm::vec3> normals(mesh.positions.size(), glm::vec3(0.0f));
    for (const auto& t : mesh.triangles) {
        const glm::vec3 n =
            glm::cross(mesh.positions[t.y] - mesh.positions[t.x], mesh.positions[t.z] - mesh.positions[t.x]);
        for (int c = 0; c < 3; ++c)
            normals[t[c]] += n;
    }
    // The texture bilinearly at a corner's coordinate: u wraps, v is clamped, texel centres at (i + 0.5) / size.
    const auto sample = [&](glm::vec2 uv) {
        const float x = uv.x * float(width) - 0.5f, y = uv.y * float(height) - 0.5f;
        const float fx = std::floor(x), fy = std::floor(y);
        const int w = int(width), h = int(height);
        const int i0 = ((int(fx) % w) + w) % w, i1 = (i0 + 1) % w;
        const int j0 = std::clamp(int(fy), 0, h - 1), j1 = std::clamp(int(fy) + 1, 0, h - 1);
        const auto texel = [&](int i, int j) { return glm::make_vec3(&texels[4 * size_t(j * w + i)]); };
        return glm::mix(glm::mix(texel(i0, j0), texel(i1, j0), x - fx), glm::mix(texel(i0, j1), texel(i1, j1), x - fx),
                        y - fy);
    };
    for (size_t t = 0; t < mesh.triangles.size(); ++t)
        for (int c = 0; c < 3; ++c) {
            const uint32_t v = mesh.triangles[t][c];
            object.mesh.push_back({glm::vec4(mesh.positions[v], 0.0f), glm::vec4(glm::normalize(normals[v]), 0.0f),
                                   glm::vec4(sample(mesh.cornerUvs[3 * t + c]), TREE_MATERIAL_CAPTURED)});
        }
    measureObject(object, mesh.positions);
    return object;
}

BonsaiObject loadBonsaiObject(const std::string& path) {
    BonsaiObject object;
    std::vector<glm::vec3> points;
    const std::string extension = std::filesystem::path(path).extension().string();
    if (extension == ".spz") {
        object.gaussians = loadGaussiansSpz(path, false);
        for (const Gaussian3D& g : object.gaussians)
            if (g.alpha > 0.5f)
                points.emplace_back(g.position[0], g.position[1], g.position[2]);
    } else if (extension == ".obj") {
        std::filesystem::path texturePath;
        const raster::TriangleMesh mesh = raster::readObj(path, &texturePath);
        uint32_t width = 1, height = 1;
        const std::vector<float> texels = texturePath.empty() ? std::vector<float>{0.6f, 0.6f, 0.6f, 1.0f}
                                                              : raster::readTexturePpm(texturePath, width, height);
        return bonsaiObjectFromMesh(mesh, texels, width, height);
    } else {
        throw std::runtime_error("loadBonsaiObject: " + path + " is neither .spz nor .obj");
    }
    measureObject(object, points);
    return object;
}

glm::mat4 bonsaiObjectTransform(const BonsaiScene& scene, const BonsaiObjectPlacement& placement) {
    const BonsaiObject& object = *placement.object;
    const float objectScale = placement.height / (object.high.y - object.low.y);
    const glm::vec3 anchor(0.5f * (object.low.x + object.high.x), object.low.y, 0.5f * (object.low.z + object.high.z));
    const glm::vec2 at = placement.position;
    return objectTransform(anchor, glm::vec3(at.x, scene.soilHeight(at.x, at.y) - 0.05f * placement.height, at.y),
                           placement.yaw, objectScale);
}

BonsaiScene makeBonsaiScene(const BonsaiSceneConfig& config) {
    BonsaiScene scene;
    scene.config = config;
    if (config.potObject) {
        // The captured pot's rim gives the outline: its length kept from the procedural pot, a rounded rectangle.
        const BonsaiObject& object = *config.potObject;
        const float potScale = config.pot.halfLength / object.rim.x;
        BonsaiPot& captured = scene.config.pot;
        captured.halfWidth = object.rim.y * potScale;
        captured.height = (object.rimHeight - object.footHeight) * potScale;
        captured.exponent = 10.0f;
        captured.wall = 0.04f;
        captured.soilHeight = captured.height - 0.04f;
    }
    const BonsaiPot& pot = scene.config.pot;
    const float inner = innerScale(pot);
    const BonsaiScale& scale = config.scale;
    // Area of the superellipse |x/a|^e + |z/b|^e <= 1: 4ab Gamma(1 + 1/e)^2 / Gamma(1 + 2/e).
    const float e = pot.exponent;
    scene.soilArea = 4.0f * pot.halfLength * inner * pot.halfWidth * inner *
                     std::pow(std::tgamma(1.0f + 1.0f / e), 2.0f) / std::tgamma(1.0f + 2.0f / e);
    const uint32_t treeCount = config.treeCount > 0 ? config.treeCount : scale.treeCount(scene.soilArea);
    // A planting up to some tens of trees; beyond, the trees follow the placement (tree_placement.hpp).
    if (treeCount > 60 && config.treeCount > 0) {
        throw std::invalid_argument("makeBonsaiScene: a planting holds at most 60 trees");
    }
    scene.placed = treeCount > 60;
    scene.lanternScale = scale.lanternUnits() / 0.28f;
    // The soil fills the pot's inner outline; the landscape's main features are about 1.7 units apart, so the pot
    // holds a few hills or mountains at every scale with relief.
    scene.terrain.outline = glm::vec4(pot.halfLength * inner, pot.halfWidth * inner, pot.exponent, 0.0f);
    // The moss patterns show real moss at kForest; at larger scales they keep their size in metres.
    scene.terrain.soil =
        glm::vec4(pot.soilHeight, pot.soilMound, 0.012f,
                  std::max(1.0f, scale.metersPerUnit() / BonsaiScale{BonsaiScale::kForest}.metersPerUnit()));
    scene.terrain.relief = glm::vec4(scale.reliefUnits(), 0.6f, scale.reliefRidges(), 0.25f);
    scene.terrain.seed = glm::uvec4(config.seed, 0u, 0u, 0u);
    scene.terrain.cover =
        glm::vec4(0.0f, scale.metersPerUnit(), BonsaiScale::kTreeLineMeters, scale.reliefExaggeration);
    scene.terrain.cover.x = calibrateForestThreshold(scene.terrain, scale.forestShare());
    const glm::vec4 glaze(pot.glaze, TREE_MATERIAL_GLAZE);
    const int around = 160;

    const bool procedural = !config.potObject;
    // Outer wall: narrower at the foot, rounded towards the rim.
    if (procedural) {
        addSurface(
            scene.mesh,
            [&](float u, float v) {
                const float scale = 0.9f + 0.1f * std::sqrt(v);
                const glm::vec2 p = potOutline(pot, 2.0f * kPi * u, scale);
                return glm::vec3(p.x, v * pot.height, p.y);
            },
            around, 8, glaze, [](glm::vec3 p) { return glm::vec3(p.x, 0.0f, p.z); });
    }
    // Rim: a rounded top from the outer to the inner edge.
    if (procedural) {
        addSurface(
            scene.mesh,
            [&](float u, float v) {
                const glm::vec2 p = potOutline(pot, 2.0f * kPi * u, 1.0f + (inner - 1.0f) * v);
                return glm::vec3(p.x, pot.height + 0.012f * std::sin(kPi * v), p.y);
            },
            around, 4, glaze, [](glm::vec3) { return glm::vec3(0.0f, 1.0f, 0.0f); });
    }
    // Inner wall down to the soil.
    BonsaiScene& s = scene;
    if (procedural) {
        addSurface(
            scene.mesh,
            [&](float u, float v) {
                const glm::vec2 p = potOutline(pot, 2.0f * kPi * u, inner);
                const float bottom = s.soilHeight(p.x, p.y) - 0.01f;
                return glm::vec3(p.x, pot.height + (bottom - pot.height) * v, p.y);
            },
            around, 2, glaze, [](glm::vec3 p) { return glm::vec3(-p.x, 0.0f, -p.z); });
    }
    // Soil with moss; a finer grid where a landscape rises from it.
    const bool relief = scene.terrain.relief.x > 0.0f;
    addSurface(
        scene.mesh,
        [&](float u, float v) {
            const glm::vec2 p = potOutline(pot, 2.0f * kPi * u, inner * v);
            return glm::vec3(p.x, s.soilHeight(p.x, p.y), p.y);
        },
        relief ? 2 * around : around, relief ? 96 : 24, glm::vec4(0.075f, 0.11f, 0.025f, TREE_MATERIAL_MOSS),
        [](glm::vec3) { return glm::vec3(0.0f, 1.0f, 0.0f); });
    // The bottom, closing the pot for views from below.
    if (procedural) {
        addSurface(
            scene.mesh,
            [&](float u, float v) {
                const glm::vec2 p = potOutline(pot, 2.0f * kPi * u, 0.9f * v);
                return glm::vec3(p.x, 0.0f, p.y);
            },
            around, 4, glaze, [](glm::vec3) { return glm::vec3(0.0f, -1.0f, 0.0f); });
    }

    // The lantern on the moss.
    scene.lanternRotation = glm::angleAxis(config.lanternYaw, glm::vec3(0.0f, 1.0f, 0.0f));
    const glm::vec3 lantern = config.lanternPosition;
    // Sunk a little, so that the edge of the captured moss disappears in the soil.
    scene.lanternTranslation = glm::vec3(lantern.x, scene.soilHeight(lantern.x, lantern.z) - 0.008f, lantern.z);
    // The fire box is about 0.15 above the lantern's ground.
    scene.lanternFlame = scene.lanternTranslation + glm::vec3(0.0f, 0.15f * scene.lanternScale, 0.0f);

    // The captured pot, its rim's long axis along x, its foot on y = 0. What lies below its foot or rises above the rim
    // (a fitted mesh's lid over the opening, stray Gaussians) is left out, and so is what faces up just below it, where
    // the soil lies.
    if (config.potObject) {
        const BonsaiObject& object = *config.potObject;
        const float potScale = pot.halfLength / object.rim.x;
        const glm::vec3 anchor(object.rimCenter.x, object.footHeight, object.rimCenter.y);
        const float margin = 0.02f * (object.rimHeight - object.footHeight);
        placeObject(
            scene, object, objectTransform(anchor, glm::vec3(0.0f), -object.rim.z, potScale), potScale,
            [&](const TreeSceneVertex* t) {
                const glm::vec3 n = glm::cross(glm::vec3(t[1].pos - t[0].pos), glm::vec3(t[2].pos - t[0].pos));
                const float y = (t[0].pos.y + t[1].pos.y + t[2].pos.y) / 3.0f;
                return y < object.rimHeight + margin && y > object.footHeight - margin &&
                       !(y > object.rimHeight - 4.0f * margin && n.y > 0.5f * glm::length(n));
            },
            [&](const Gaussian3D& g) {
                return g.position[1] < object.rimHeight + margin && g.position[1] > object.footHeight - margin;
            });
    }
    // The objects on the soil, sunk a little so that their unseen undersides disappear in it.
    for (const BonsaiObjectPlacement& placement : config.objects) {
        const BonsaiObject& object = *placement.object;
        placeObject(scene, object, bonsaiObjectTransform(scene, placement),
                    placement.height / (object.high.y - object.low.y));
    }

    // Trees: a tall main tree off-centre at the back, the others around it,
    // smaller and leaning slightly outwards the farther away they stand.
    const float style = config.style >= 0.0f ? std::min(config.style, 1.0f) : scale.bonsaiStyle();
    scene.types = {mixTreeTypes(tree_presets::bonsaiBroadleaf(),
                                tree_presets::bonsaiStyled(tree_presets::bonsaiBroadleaf()), style),
                   mixTreeTypes(tree_presets::bonsaiPine(),
                                tree_presets::bonsaiPinePads(tree_presets::bonsaiStyled(tree_presets::bonsaiPine())),
                                style)};
    if (scene.placed) {
        // The trees nearest to the middle of the soil, until a camera asks for its own.
        scene.placement = makeTreePlacement(scene, config.seed);
        const glm::vec3 middle(0.0f, scene.soilHeight(0.0f, 0.0f), 0.0f);
        for (const TreeSite& site : nearestTreeSites(scene.terrain, scene.placement, middle, config.nearTrees)) {
            scene.trees.push_back(treeInstanceFor(site, scene.types, config.seed));
        }
        scene.focus = glm::vec3(0.0f, pot.soilHeight + 0.54f, 0.0f); // above the soil, among the trees
        return scene;
    }
    // A single tree stands near the middle; in a planting the main tree is off-centre at the back.
    const float planting = std::clamp(float(treeCount - 1) / 12.0f, 0.0f, 1.0f);
    const glm::vec2 main = config.treeSpread * glm::mix(glm::vec2(-0.2f, -0.05f), glm::vec2(-0.4f, -0.15f), planting);
    // Trees about 1.9 m apart at least.
    const float minDistance = 1.9f / scale.metersPerUnit();
    // Trees grow in the forest only (all of the soil at small scales).
    auto forested = [&](glm::vec2 p) { return tree::treeLandCover(scene.terrain, p.x, p.y).vegetation.x > 0.5f; };
    std::vector<glm::vec2> positions;
    if (forested(main)) {
        positions.push_back(main);
    }
    for (uint32_t key = 0; positions.size() < treeCount && key < 20000; ++key) {
        const glm::vec2 p((2.0f * hashUnit(config.seed, 2 * key) - 1.0f) * pot.halfLength * config.treeSpread,
                          (2.0f * hashUnit(config.seed, 2 * key + 1) - 1.0f) * pot.halfWidth * config.treeSpread);
        if (scene.potMeasure(p.x, p.y, inner * 0.82f) > 1.0f || !forested(p)) {
            continue;
        }
        if (glm::length(p - glm::vec2(lantern.x, lantern.z)) < 0.1f + 1.0f * scale.lanternUnits()) {
            continue;
        }
        bool free = true;
        for (const glm::vec2& q : positions) {
            free = free && glm::length(p - q) >= minDistance;
        }
        if (free) {
            positions.push_back(p);
        }
    }
    const glm::vec4 up = treeUpRotation(glm::vec3(0.0f, 1.0f, 0.0f));
    const glm::quat upRotation(up.w, up.x, up.y, up.z);
    for (uint32_t i = 0; i < positions.size(); ++i) {
        const glm::vec2 p = positions[i];
        const float distance = glm::length(p - main) / (2.0f * pot.halfLength);
        const float frontFactor = 1.0f - 0.25f * std::max(p.y / pot.halfWidth, 0.0f);
        // The tallest tree at the scale's height; 0.13 is that at kForest, where the bonsai presets are 13 m.
        const float tallest = 0.13f * scale.treeHeightUnits() / BonsaiScale{BonsaiScale::kForest}.treeHeightUnits();
        float treeScale = config.treeSize * tallest * (1.0f - 0.55f * distance) * frontFactor *
                          (0.85f + 0.3f * hashUnit(config.seed, 9000 + i));
        const uint32_t type = config.species == BonsaiSpecies::Pine        ? 1u
                              : config.species == BonsaiSpecies::Broadleaf ? 0u
                              : (i % 4 == 3)                               ? 1u
                                                                           : 0u;
        if (type == 1u) {
            treeScale *= 0.9f;
        }
        const float yaw = 2.0f * kPi * hashUnit(config.seed, 9100 + i);
        glm::quat rotation = glm::angleAxis(yaw, glm::vec3(0.0f, 1.0f, 0.0f)) * upRotation;
        const glm::vec2 outwards = p - main;
        if (glm::length(outwards) > 1e-3f) {
            const glm::vec2 d = glm::normalize(outwards);
            const float lean = glm::radians(9.0f) * std::min(distance * 2.0f, 1.0f);
            rotation = glm::angleAxis(lean, glm::vec3(d.y, 0.0f, -d.x)) * rotation;
        }
        TreeInstance inst{};
        inst.posScale = glm::vec4(p.x, scene.soilHeight(p.x, p.y) - 0.01f, p.y, treeScale);
        inst.rot = glm::vec4(rotation.x, rotation.y, rotation.z, rotation.w);
        inst.ids = glm::uvec4(type, config.seed * 977u + i, 0u, 0u);
        scene.trees.push_back(inst);
    }
    scene.focus = glm::vec3(0.0f, pot.soilHeight + 0.54f, 0.0f); // above the soil, among the trees
    return scene;
}

std::vector<TreeSceneVertex> sampleMeshSurfels(const std::vector<TreeSceneVertex>& mesh, float spacing, uint32_t seed) {
    std::vector<TreeSceneVertex> surfels;
    for (size_t t = 0; t + 2 < mesh.size(); t += 3) {
        const TreeSceneVertex& a = mesh[t];
        const TreeSceneVertex& b = mesh[t + 1];
        const TreeSceneVertex& c = mesh[t + 2];
        const glm::vec3 pa(a.pos), pb(b.pos), pc(c.pos);
        const float area = 0.5f * glm::length(glm::cross(pb - pa, pc - pa));
        const uint32_t key = uint32_t(t) * 4u;
        const float wanted = area / (spacing * spacing);
        const uint32_t count = uint32_t(std::floor(wanted + tree::treeRand(seed, key)));
        for (uint32_t i = 0; i < count; ++i) {
            float r1 = tree::treeRand(tree::treeHash2(seed, key + 1), i);
            float r2 = tree::treeRand(tree::treeHash2(seed, key + 2), i);
            if (r1 + r2 > 1.0f) {
                r1 = 1.0f - r1;
                r2 = 1.0f - r2;
            }
            const float r0 = 1.0f - r1 - r2;
            TreeSceneVertex surfel{};
            surfel.pos = glm::vec4(r0 * pa + r1 * pb + r2 * pc, spacing);
            surfel.normal = glm::vec4(
                glm::normalize(r0 * glm::vec3(a.normal) + r1 * glm::vec3(b.normal) + r2 * glm::vec3(c.normal)), 0.0f);
            surfel.color = a.color;
            surfels.push_back(surfel);
        }
    }
    return surfels;
}

std::vector<MossPatch> extractMossPatches(const std::vector<Gaussian3D>& capture, uint32_t sectors, float inner,
                                          float outer, float maxHeight, float minGreen) {
    std::vector<MossPatch> patches(sectors);
    std::vector<uint32_t> green(sectors, 0);
    const float sector = 2.0f * kPi / float(sectors);
    const float fade = 0.02f; // width of the fade at the cuts
    for (const Gaussian3D& g : capture) {
        const glm::vec2 p(g.position[0], g.position[2]);
        const float r = glm::length(p);
        if (r < inner || r > outer || g.position[1] > maxHeight || g.position[1] < -maxHeight) {
            continue;
        }
        float angle = std::atan2(p.y, p.x);
        if (angle < 0.0f) {
            angle += 2.0f * kPi;
        }
        const uint32_t s = std::min(uint32_t(angle / sector), sectors - 1);
        // Distance to the nearest cut, radially and around.
        const float around = std::min(angle - float(s) * sector, float(s + 1) * sector - angle) * r;
        const float radial = std::min(r - inner, outer - r);
        const float weight = std::clamp(std::min(around, radial) / fade, 0.0f, 1.0f);
        if (weight <= 0.0f) {
            continue;
        }
        Gaussian3D h = g;
        h.alpha *= weight * weight * (3.0f - 2.0f * weight);
        patches[s].push_back(h);
        // The degree 0 colour coefficients order like the colours.
        green[s] += g.color[1] > g.color[0] && g.color[1] > g.color[2] ? 1 : 0;
    }
    for (uint32_t s = 0; s < sectors; ++s) {
        if (float(green[s]) < minGreen * float(patches[s].size())) {
            patches[s].clear();
        }
    }
    for (MossPatch& patch : patches) {
        if (patch.empty()) {
            continue;
        }
        // Centre on the patch's middle; the ground is the median height.
        glm::vec2 centre(0.0f);
        std::vector<float> heights;
        for (const Gaussian3D& g : patch) {
            centre += glm::vec2(g.position[0], g.position[2]);
            heights.push_back(g.position[1]);
        }
        centre /= float(patch.size());
        std::nth_element(heights.begin(), heights.begin() + heights.size() / 2, heights.end());
        const float ground = heights[heights.size() / 2];
        for (Gaussian3D& g : patch) {
            g.position[0] -= centre.x;
            g.position[1] -= ground;
            g.position[2] -= centre.y;
        }
    }
    patches.erase(std::remove_if(patches.begin(), patches.end(), [](const MossPatch& p) { return p.empty(); }),
                  patches.end());
    return patches;
}

std::vector<TreeLodInstance> placeMossPatches(const BonsaiScene& scene, uint32_t patchCount, uint32_t count,
                                              uint32_t seed, float scale) {
    std::vector<TreeLodInstance> instances;
    if (patchCount == 0) {
        return instances;
    }
    const BonsaiPot& pot = scene.config.pot;
    const float inner = innerScale(pot);
    const glm::vec2 lantern(scene.lanternTranslation.x, scene.lanternTranslation.z);
    for (uint32_t key = 0; instances.size() < count && key < 100 * count; ++key) {
        const glm::vec2 p((2.0f * hashUnit(seed, 3 * key) - 1.0f) * pot.halfLength,
                          (2.0f * hashUnit(seed, 3 * key + 1) - 1.0f) * pot.halfWidth);
        if (scene.potMeasure(p.x, p.y, inner * 0.88f) > 1.0f || glm::length(p - lantern) < 0.3f) {
            continue;
        }
        // Moss covers the forest floor, not open land or rock.
        if (tree::treeLandCover(scene.terrain, p.x, p.y).vegetation.x < 0.5f) {
            continue;
        }
        // The plane that touches the soil at the patch's centre, a little above it.
        const float e = 0.02f;
        TreeLodInstance instance{};
        instance.posYaw =
            glm::vec4(p.x, scene.soilHeight(p.x, p.y) + 0.002f, p.y, 2.0f * kPi * hashUnit(seed, 3 * key + 2));
        instance.plane =
            glm::vec4((scene.soilHeight(p.x + e, p.y) - scene.soilHeight(p.x - e, p.y)) / (2.0f * e),
                      (scene.soilHeight(p.x, p.y + e) - scene.soilHeight(p.x, p.y - e)) / (2.0f * e), scale, 0.0f);
        instance.ids = glm::uvec4(tree::treeHash2(seed, key) % patchCount, 0u, 0u, 0u);
        instances.push_back(instance);
    }
    return instances;
}

std::vector<Gaussian3D> scatterMossPatches(const std::vector<MossPatch>& patches,
                                           const std::vector<TreeLodInstance>& instances) {
    std::vector<Gaussian3D> result;
    for (const TreeLodInstance& inst : instances) {
        const glm::vec4 turn = tree::qaxis(glm::vec3(0.0f, 1.0f, 0.0f), inst.posYaw.w);
        const float scale = inst.plane.z;
        for (const Gaussian3D& g : patches.at(inst.ids.x)) {
            const glm::vec3 local = tree::qrot(turn, scale * glm::vec3(g.position[0], 0.0f, g.position[2]));
            glm::vec3 pos = glm::vec3(inst.posYaw) + local;
            pos.y += inst.plane.x * local.x + inst.plane.y * local.z + scale * g.position[1];
            const glm::vec4 rot =
                glm::normalize(tree::qmul(turn, glm::vec4(g.rotation[0], g.rotation[1], g.rotation[2], g.rotation[3])));
            Gaussian3D h = g;
            h.position = {pos.x, pos.y, pos.z};
            h.rotation = {rot.x, rot.y, rot.z, rot.w};
            h.scale = {scale * g.scale[0], scale * g.scale[1], scale * g.scale[2]};
            result.push_back(h);
        }
    }
    return result;
}

TreeTypeParams mixTreeTypes(const TreeTypeParams& a, const TreeTypeParams& b, float t) {
    // The parameters are vec4s only.
    constexpr size_t kCount = sizeof(TreeTypeParams) / sizeof(glm::vec4);
    TreeTypeParams result;
    const glm::vec4* pa = reinterpret_cast<const glm::vec4*>(&a);
    const glm::vec4* pb = reinterpret_cast<const glm::vec4*>(&b);
    glm::vec4* pr = reinterpret_cast<glm::vec4*>(&result);
    for (size_t i = 0; i < kCount; ++i) {
        pr[i] = glm::mix(pa[i], pb[i], t);
    }
    return result;
}

namespace tree_presets {

TreeTypeParams bonsaiStyled(const TreeTypeParams& natural) {
    TreeTypeParams p = natural;
    p.shapeBase.x = 2.0f;  // hemispherical crown
    p.shapeBase.y = 0.35f; // branches start a third up
    p.shapeBase.w = 0.0f;  // no random height
    p.ratios.y *= 2.6f;    // a thick trunk
    p.ratios.w = 1.2f;     // flaring base
    p.trunk.x = 6.0f;      // spreading surface roots (nebari)
    p.trunk.y = 0.8f;
    p.nCurveRes.x = 8.0f; // an S-curved trunk (moyogi)
    p.nCurve.x = 35.0f;
    p.nCurveBack.x = -45.0f;
    p.nCurveV.x = 10.0f;
    p.nTaper.x = 0.85f;
    p.nLength.y *= 1.7f; // long, nearly horizontal branches: a wide, flat crown
    p.nDownAngle.y = 80.0f;
    p.nDownAngleV.y = -25.0f;
    p.nBranches.y *= 0.7f;
    p.nCurve.y = 15.0f;    // branch tips turning up
    p.nBranches.z *= 1.4f; // dense foliage pads
    p.leafCount.x *= 1.5f;
    p.leafCount.y *= 1.4f; // leaves large for the tree's size
    return p;
}

TreeTypeParams bonsaiPinePads(const TreeTypeParams& styled) {
    TreeTypeParams p = styled;
    p.nBranches.y = 22.0f; // few main branches
    p.nBranches.z = 42.0f; // many short twigs, spreading flat from them: dense pads at the branch ends
    p.nLength.z = 0.16f;
    p.nDownAngle.z = 75.0f;
    p.nDownAngleV.z = 10.0f;
    p.nCurve.z = 20.0f; // twig tips turning up
    p.leafCount.x = 220.0f;
    p.leafCount.y = 0.2f; // long needles, and wide tufts of them
    p.leafCount.z = 0.18f;
    p.leafColor = glm::vec4(0.035f, 0.07f, 0.03f, 0.0f); // the dark green of old pine needles
    p.leafColorFall = p.leafColor;
    return p;
}

TreeTypeParams bonsaiBroadleaf() {
    TreeTypeParams p = quakingAspen();
    p.shapeBase = glm::vec4(2.0f, 0.25f, 13.0f, 2.0f);
    p.ratios = glm::vec4(3.0f, 0.022f, 1.25f, 0.4f);
    p.leafCount = glm::vec4(28.0f, 0.15f, 0.7f, 0.4f);
    p.leafShape = glm::vec4(55.0f, 0.0f, 35.0f, 0.42f);
    p.leafLobes = glm::vec4(1.0f, 0.0f, 1.0f, 0.02f);
    p.leafColor = glm::vec4(0.08f, 0.15f, 0.035f, 0.0f);
    p.barkColor = glm::vec4(0.17f, 0.14f, 0.11f, 0.0f);
    p.leafColorFall = glm::vec4(0.3f, 0.11f, 0.02f, 0.0f);
    p.trunk = glm::vec4(5.0f, 0.45f, 30.0f, 1.0f);
    p.nCurveRes = glm::vec4(6.0f, 5.0f, 3.0f, 1.0f);
    p.nCurveV = glm::vec4(25.0f, 50.0f, 75.0f, 0.0f);
    p.nSegSplits = glm::vec4(0.0f, 0.15f, 0.0f, 0.0f);
    p.nSplitAngle = glm::vec4(0.0f, 25.0f, 0.0f, 0.0f);
    p.nBranches = glm::vec4(1.0f, 40.0f, 22.0f, 10.0f);
    return p;
}

TreeTypeParams bonsaiPine() {
    TreeTypeParams p = pine();
    p.ratios = glm::vec4(3.0f, 0.02f, 1.4f, 0.5f);
    p.leafCount = glm::vec4(70.0f, 0.14f, 0.07f, 0.2f);
    p.leafColor = glm::vec4(0.04f, 0.085f, 0.04f, 0.0f);
    p.barkColor = glm::vec4(0.2f, 0.11f, 0.07f, 0.0f);
    p.trunk = glm::vec4(4.0f, 0.3f, 20.0f, 0.0f);
    p.nBranches = glm::vec4(1.0f, 45.0f, 16.0f, 0.0f);
    return p;
}

} // namespace tree_presets

} // namespace klartraum
