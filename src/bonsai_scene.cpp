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
        around, 24, glm::vec4(0.075f, 0.11f, 0.025f, TREE_MATERIAL_MOSS),
        [](glm::vec3) { return glm::vec3(0.0f, 1.0f, 0.0f); });
    // The bottom, closing the pot for views from below.
    addSurface(
        scene.mesh,
        [&](float u, float v) {
            const glm::vec2 p = potOutline(pot, 2.0f * kPi * u, 0.9f * v);
            return glm::vec3(p.x, 0.0f, p.y);
        },
        around, 4, glaze, [](glm::vec3) { return glm::vec3(0.0f, -1.0f, 0.0f); });

    // The lantern on the moss.
    scene.lanternRotation = glm::angleAxis(config.lanternYaw, glm::vec3(0.0f, 1.0f, 0.0f));
    const glm::vec3 lantern = config.lanternPosition;
    // Sunk a little, so that the edge of the captured moss disappears in the soil.
    scene.lanternTranslation = glm::vec3(lantern.x, scene.soilHeight(lantern.x, lantern.z) - 0.008f, lantern.z);
    // The fire box is about 0.15 above the lantern's ground.
    scene.lanternFlame = scene.lanternTranslation + glm::vec3(0.0f, 0.15f * config.lanternScale, 0.0f);

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

namespace tree_presets {

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
