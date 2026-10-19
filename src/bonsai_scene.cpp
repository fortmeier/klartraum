// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/tree/bonsai_scene.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
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

} // namespace

float BonsaiScene::potMeasure(float x, float z, float scale) const {
    const BonsaiPot& pot = config.pot;
    return std::pow(std::abs(x / (pot.halfLength * scale)), pot.exponent) +
           std::pow(std::abs(z / (pot.halfWidth * scale)), pot.exponent);
}

float BonsaiScene::soilHeight(float x, float z) const {
    const BonsaiPot& pot = config.pot;
    const float r = std::min(std::pow(potMeasure(x, z, innerScale(pot)), 1.0f / pot.exponent), 1.0f);
    const float bumps = 0.012f * (tree::treeFbm(glm::vec3(x * 3.0f, 0.0f, z * 3.0f)) - 0.5f);
    return pot.soilHeight + pot.soilMound * (1.0f - r * r) + bumps * (1.0f - r);
}

BonsaiScene makeBonsaiScene(const BonsaiSceneConfig& config) {
    if (config.treeCount < 1 || config.treeCount > 20) {
        throw std::invalid_argument("makeBonsaiScene: treeCount must be in [1, 20]");
    }
    BonsaiScene scene;
    scene.config = config;
    const BonsaiPot& pot = config.pot;
    const float inner = innerScale(pot);
    const glm::vec4 glaze(pot.glaze, TREE_MATERIAL_GLAZE);
    const int around = 160;

    // Outer wall: narrower at the foot, rounded towards the rim.
    addSurface(
        scene.mesh,
        [&](float u, float v) {
            const float scale = 0.9f + 0.1f * std::sqrt(v);
            const glm::vec2 p = potOutline(pot, 2.0f * kPi * u, scale);
            return glm::vec3(p.x, v * pot.height, p.y);
        },
        around, 8, glaze, [](glm::vec3 p) { return glm::vec3(p.x, 0.0f, p.z); });
    // Rim: a rounded top from the outer to the inner edge.
    addSurface(
        scene.mesh,
        [&](float u, float v) {
            const glm::vec2 p = potOutline(pot, 2.0f * kPi * u, 1.0f + (inner - 1.0f) * v);
            return glm::vec3(p.x, pot.height + 0.012f * std::sin(kPi * v), p.y);
        },
        around, 4, glaze, [](glm::vec3) { return glm::vec3(0.0f, 1.0f, 0.0f); });
    // Inner wall down to the soil.
    BonsaiScene& s = scene;
    addSurface(
        scene.mesh,
        [&](float u, float v) {
            const glm::vec2 p = potOutline(pot, 2.0f * kPi * u, inner);
            const float bottom = s.soilHeight(p.x, p.y) - 0.01f;
            return glm::vec3(p.x, pot.height + (bottom - pot.height) * v, p.y);
        },
        around, 2, glaze, [](glm::vec3 p) { return glm::vec3(-p.x, 0.0f, -p.z); });
    // Soil with moss.
    addSurface(
        scene.mesh,
        [&](float u, float v) {
            const glm::vec2 p = potOutline(pot, 2.0f * kPi * u, inner * v);
            return glm::vec3(p.x, s.soilHeight(p.x, p.y), p.y);
        },
        around, 24, glm::vec4(0.2f, 0.33f, 0.1f, TREE_MATERIAL_MOSS),
        [](glm::vec3) { return glm::vec3(0.0f, 1.0f, 0.0f); });
    // The table.
    const float t = config.tableHalfSize;
    addSurface(
        scene.mesh,
        [&](float u, float v) { return glm::vec3((2.0f * u - 1.0f) * t, 0.0f, (2.0f * v - 1.0f) * t * 0.6f); }, 1, 1,
        glm::vec4(0.23f, 0.15f, 0.09f, TREE_MATERIAL_WOOD), [](glm::vec3) { return glm::vec3(0.0f, 1.0f, 0.0f); });

    // The lantern on the moss.
    scene.lanternRotation = glm::angleAxis(config.lanternYaw, glm::vec3(0.0f, 1.0f, 0.0f));
    const glm::vec3 lantern = config.lanternPosition;
    // Sunk a little, so that the edge of the captured moss disappears in the soil.
    scene.lanternTranslation = glm::vec3(lantern.x, scene.soilHeight(lantern.x, lantern.z) - 0.008f, lantern.z);

    // Trees: a tall main tree off-centre at the back, the others around it,
    // smaller and leaning slightly outwards the farther away they stand.
    scene.types = {tree_presets::bonsaiBroadleaf(), tree_presets::bonsaiPine()};
    const glm::vec2 main(-0.4f, -0.15f);
    const float minDistance = 0.3f;
    std::vector<glm::vec2> positions = {main};
    for (uint32_t key = 0; positions.size() < config.treeCount && key < 20000; ++key) {
        const glm::vec2 p((2.0f * hashUnit(config.seed, 2 * key) - 1.0f) * pot.halfLength,
                          (2.0f * hashUnit(config.seed, 2 * key + 1) - 1.0f) * pot.halfWidth);
        if (scene.potMeasure(p.x, p.y, inner * 0.82f) > 1.0f) {
            continue;
        }
        if (glm::length(p - glm::vec2(lantern.x, lantern.z)) < 0.38f) {
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
        float scale =
            0.13f * (1.0f - 0.55f * distance) * frontFactor * (0.85f + 0.3f * hashUnit(config.seed, 9000 + i));
        const uint32_t type = (i % 4 == 3) ? 1u : 0u;
        if (type == 1u) {
            scale *= 0.9f;
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
        inst.posScale = glm::vec4(p.x, scene.soilHeight(p.x, p.y) - 0.01f, p.y, scale);
        inst.rot = glm::vec4(rotation.x, rotation.y, rotation.z, rotation.w);
        inst.ids = glm::uvec4(type, config.seed * 977u + i, 0u, 0u);
        scene.trees.push_back(inst);
    }
    scene.focus = glm::vec3(0.0f, 0.8f, 0.0f);
    return scene;
}

std::vector<Gaussian3D> sampleMeshGaussians(const std::vector<TreeSceneVertex>& mesh, float spacing,
                                            const TreeRenderSettings& lighting, uint32_t seed) {
    const float shC0 = 0.28209479f;
    std::vector<Gaussian3D> gaussians;
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
            const glm::vec3 pos = r0 * pa + r1 * pb + r2 * pc;
            const glm::vec3 normal =
                glm::normalize(r0 * glm::vec3(a.normal) + r1 * glm::vec3(b.normal) + r2 * glm::vec3(c.normal));
            const glm::vec4 color = a.color;
            const glm::vec3 shaded =
                tree::treeDisplayColor(tree::treeShade(tree::treeSceneAlbedo(color, pos), normal, lighting));

            // Thin axis (z) along the normal, spun randomly about it.
            const glm::vec4 a4 = treeUpRotation(normal);
            const glm::quat align(a4.w, a4.x, a4.y, a4.z);
            const float spin = 2.0f * kPi * tree::treeRand(tree::treeHash2(seed, key + 3), i);
            const glm::quat q = align * glm::angleAxis(spin, glm::vec3(0.0f, 0.0f, 1.0f));

            Gaussian3D g{};
            g.position = {pos.x, pos.y, pos.z};
            g.rotation = {q.x, q.y, q.z, q.w};
            g.scale = {spacing, spacing, 0.08f * spacing};
            g.color = {(shaded.x - 0.5f) / shC0, (shaded.y - 0.5f) / shC0, (shaded.z - 0.5f) / shC0};
            g.alpha = 0.95f;
            gaussians.push_back(g);
        }
    }
    return gaussians;
}

namespace tree_presets {

TreeTypeParams bonsaiBroadleaf() {
    TreeTypeParams p = quakingAspen();
    p.shapeBase = glm::vec4(2.0f, 0.25f, 13.0f, 2.0f);
    p.ratios = glm::vec4(3.0f, 0.022f, 1.25f, 0.8f);
    p.leafCount = glm::vec4(28.0f, 0.15f, 0.7f, 0.4f);
    p.leafShape = glm::vec4(55.0f, 0.0f, 35.0f, 0.42f);
    p.leafLobes = glm::vec4(1.0f, 0.0f, 1.0f, 0.02f);
    p.leafColor = glm::vec4(0.22f, 0.42f, 0.11f, 0.0f);
    p.barkColor = glm::vec4(0.36f, 0.31f, 0.27f, 0.0f);
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
    p.leafColor = glm::vec4(0.1f, 0.25f, 0.12f, 0.0f);
    p.nBranches = glm::vec4(1.0f, 45.0f, 16.0f, 0.0f);
    return p;
}

} // namespace tree_presets

} // namespace klartraum
