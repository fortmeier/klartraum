// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/tree/tree_library.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

#include <glm/gtc/quaternion.hpp>

namespace klartraum {

namespace {

// A Gaussian of a tree with its albedo as colour: the copies are lit every frame (tree_lod_write.comp).
Gaussian3D albedoGaussian(glm::vec3 pos, glm::vec4 rot, glm::vec3 scale, glm::vec3 albedo, float alpha) {
    Gaussian3D g{};
    g.position = {pos.x, pos.y, pos.z};
    g.rotation = {rot.x, rot.y, rot.z, rot.w};
    g.scale = {scale.x, scale.y, scale.z};
    g.color = {albedo.x, albedo.y, albedo.z};
    g.alpha = alpha;
    return g;
}

// The eigenvalues (ascending) and unit eigenvectors of the symmetric 2 x 2 matrix [[a, b], [b, d]].
void symmetricEigen2(float a, float b, float d, glm::vec2& values, glm::vec2& major) {
    const float mean = 0.5f * (a + d);
    const float spread = std::sqrt(std::max(0.25f * (a - d) * (a - d) + b * b, 0.0f));
    values = glm::vec2(mean - spread, mean + spread);
    major = std::abs(b) > 1e-12f ? glm::normalize(glm::vec2(b, values.y - a))
                                 : (a >= d ? glm::vec2(1.0f, 0.0f) : glm::vec2(0.0f, 1.0f));
}

// The Gaussians along a stem segment: `count` of them, each as thick as the stem and as long as its share.
void addStem(std::vector<Gaussian3D>& out, const SegmentRecord& s, int count, glm::vec3 bark) {
    const glm::vec3 a(s.fromPosRadius), b(s.toPosRadius);
    const float radius = 0.5f * (s.fromPosRadius.w + s.toPosRadius.w);
    const float length = glm::length(b - a);
    const glm::vec4 rot = tree::qnlerp(s.fromRot, s.toRot, 0.5f);
    for (int k = 0; k < count; ++k) {
        const glm::vec3 c = glm::mix(a, b, (k + 0.5f) / count);
        out.push_back(albedoGaussian(c, rot, glm::vec3(radius, radius, 0.6f * length / count), bark, 0.95f));
    }
}

} // namespace

GaussianLods buildTreeGaussianLevels(const TreeRecords& records, const TreeTypeParams& type,
                                     std::array<uint32_t, 4> budgets) {
    float height = 0.0f;
    for (const SegmentRecord& s : records.segments) {
        height = std::max(height, std::max(s.fromPosRadius.y, s.toPosRadius.y));
    }
    // The leaves: their middles, areas and albedos.
    struct Leaf {
        glm::vec3 center;
        float area;
        glm::vec3 albedo;
        float length;
    };
    std::vector<Leaf> leaves;
    // What a leaf covers as the near trees draw it (tree_gaussians.comp, 8 Gaussians of opacity 0.9).
    const float drawn = tree::treeLeafGaussianCoverage(type, 8u, 0.9f);
    glm::vec3 low(1e30f), high(-1e30f), crown(0.0f);
    float total = 0.0f;
    for (const LeafRecord& l : records.leaves) {
        const float length = l.posScale.w;
        if (length <= 0.0f) {
            continue;
        }
        Leaf leaf;
        leaf.center = glm::vec3(l.posScale) + 0.5f * length * tree::qforward(l.rot);
        leaf.area = drawn * length * length;
        leaf.albedo = tree::treeLeafSeasonAlbedo(type, l.ids.y, 1u, l.ids.w);
        leaf.length = length;
        leaves.push_back(leaf);
        low = glm::min(low, leaf.center);
        high = glm::max(high, leaf.center);
        crown += leaf.area * leaf.center;
        total += leaf.area;
        height = std::max(height, l.posScale.y + length);
    }
    crown = total > 0.0f ? crown / total : glm::vec3(0.0f, 0.6f * height, 0.0f);
    // The stems, thickest first.
    std::vector<size_t> stems(records.segments.size());
    std::iota(stems.begin(), stems.end(), 0);
    auto radiusOf = [&](size_t i) {
        return 0.5f * (records.segments[i].fromPosRadius.w + records.segments[i].toPosRadius.w);
    };
    std::sort(stems.begin(), stems.end(), [&](size_t a, size_t b) { return radiusOf(a) > radiusOf(b); });

    GaussianLods lods;
    for (int level = 0; level < 4; ++level) {
        std::vector<Gaussian3D>& out = lods.levels[level];
        const uint32_t budget = budgets[level];
        // A quarter of the budget for the stems thicker than a thousandth of the height, fewer and longer
        // Gaussians on coarser levels.
        const uint32_t stemBudget = leaves.empty() ? budget : budget / 4;
        for (size_t i : stems) {
            const SegmentRecord& s = records.segments[i];
            const float radius = radiusOf(i);
            const float length = glm::length(glm::vec3(s.toPosRadius) - glm::vec3(s.fromPosRadius));
            if (radius < 1e-3f * height || length <= 0.0f) {
                break;
            }
            const int count = std::clamp(int(std::ceil(length / (2.0f * radius * float(1 << level)))), 1, 8);
            if (out.size() + count > stemBudget) {
                break;
            }
            addStem(out, s, count, glm::vec3(type.barkColor));
        }
        if (leaves.empty()) {
            continue;
        }
        // The leaves in the cells of the finest grid whose occupied cells fit the rest of the budget.
        const uint32_t clusters = std::max<uint32_t>(budget - uint32_t(out.size()), 1u);
        const float size = std::max(std::max(high.x - low.x, high.y - low.y), std::max(high.z - low.z, 1e-6f));
        auto cellsOf = [&](int n, std::unordered_map<uint64_t, std::vector<uint32_t>>& cells) {
            cells.clear();
            for (uint32_t i = 0; i < leaves.size(); ++i) {
                const glm::ivec3 c = glm::clamp(glm::ivec3((leaves[i].center - low) / size * float(n)), 0, n - 1);
                cells[(uint64_t(c.x) * uint64_t(n) + uint64_t(c.y)) * uint64_t(n) + uint64_t(c.z)].push_back(i);
            }
        };
        std::unordered_map<uint64_t, std::vector<uint32_t>> cells;
        int lo = 1, hi = 1024;
        while (lo + 1 < hi) {
            const int mid = (lo + hi) / 2;
            cellsOf(mid, cells);
            if (cells.size() <= clusters) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        cellsOf(lo, cells);
        std::vector<uint64_t> keys;
        for (const auto& [key, members] : cells) {
            keys.push_back(key);
        }
        std::sort(keys.begin(), keys.end());
        for (uint64_t key : keys) {
            const std::vector<uint32_t>& members = cells[key];
            float area = 0.0f, length = 0.0f;
            glm::vec3 mean(0.0f), albedo(0.0f);
            for (uint32_t i : members) {
                area += leaves[i].area;
                mean += leaves[i].area * leaves[i].center;
                albedo += leaves[i].area * leaves[i].albedo;
                length += leaves[i].area * leaves[i].length;
            }
            mean /= area;
            albedo /= area;
            length /= area;
            // The spread of the leaves, and of a leaf itself.
            glm::mat3 cov(length * length / 12.0f);
            for (uint32_t i : members) {
                const glm::vec3 d = leaves[i].center - mean;
                cov += (leaves[i].area / area) * glm::outerProduct(d, d);
            }
            // Flat towards the outside of the crown, which also gives the normal its lighting.
            glm::vec3 normal = mean - crown;
            normal = glm::length(normal) > 1e-6f ? glm::normalize(normal) : glm::vec3(0.0f, 1.0f, 0.0f);
            const glm::vec3 helper =
                std::abs(normal.y) < 0.9f ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
            const glm::vec3 u = glm::normalize(glm::cross(helper, normal));
            const glm::vec3 v = glm::cross(normal, u);
            glm::vec2 values, major;
            symmetricEigen2(glm::dot(u, cov * u), glm::dot(u, cov * v), glm::dot(v, cov * v), values, major);
            const glm::vec3 t1 = major.x * u + major.y * v;
            const glm::vec3 t2 = glm::cross(normal, t1);
            // Wider than the leaves' spread, so that neighbouring cells' Gaussians close the crown without gaps.
            const float spread = 1.6f;
            float s1 = spread * std::sqrt(std::max(values.y, 1e-12f));
            float s2 = std::max(spread * std::sqrt(std::max(values.x, 1e-12f)), s1 / 3.0f);
            const float sn = std::max(0.5f * std::sqrt(std::max(glm::dot(normal, cov * normal), 1e-12f)), s2 / 2.0f);
            // As opaque as the leaves cover the Gaussian's face: half their drawn area (turned every way) over its
            // own.
            const float alpha =
                std::clamp(1.0f - std::exp(-3.0f * area / (2.0f * tree::TREE_PI * s1 * s2)), 0.02f, 0.98f);
            const glm::quat q = glm::quat_cast(glm::mat3(t1, t2, normal));
            out.push_back(albedoGaussian(mean, glm::vec4(q.x, q.y, q.z, q.w), glm::vec3(s1, s2, sn), albedo, alpha));
        }
    }
    return lods;
}

std::vector<GaussianLods> buildTreeGaussianLibraryLevels(const std::vector<TreeTypeParams>& types,
                                                         const std::vector<float>& seasons, std::vector<float>& heights,
                                                         uint32_t variants, std::array<uint32_t, 4> budgets) {
    if (types.empty() || seasons.empty()) {
        throw std::invalid_argument("buildTreeGaussianLibraryLevels: needs tree types and seasons");
    }
    const glm::vec4 up = treeUpRotation(glm::vec3(0.0f, 1.0f, 0.0f));
    std::vector<GaussianLods> library;
    heights.assign(types.size(), 0.0f);
    for (uint32_t tv = 0; tv < types.size() * variants; ++tv) {
        const uint32_t type = tv / variants;
        for (float season : seasons) {
            TreeFrame frame = defaultTreeFrame();
            frame.wind.w = 0.0f;
            frame.timeSeason.y = season;
            TreeInstance inst{};
            inst.posScale = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
            inst.rot = up;
            inst.ids = glm::uvec4(type, 1u + tv % variants, 0u, 0u);
            const TreeRecords records = generateTreesReference(types, {inst}, frame);
            for (const SegmentRecord& s : records.segments) {
                heights[type] = std::max(heights[type], std::max(s.fromPosRadius.y, s.toPosRadius.y));
            }
            library.push_back(buildTreeGaussianLevels(records, types[type], budgets));
        }
    }
    return library;
}

std::vector<MossPatch> buildTreeGaussianLibrary(const std::vector<TreeTypeParams>& types,
                                                const std::vector<float>& seasons, std::vector<float>& heights,
                                                bool reduced, uint32_t variants) {
    if (types.empty() || seasons.empty()) {
        throw std::invalid_argument("buildTreeGaussianLibrary: needs tree types and seasons");
    }
    std::vector<TreeTypeParams> grown = types;
    if (reduced) {
        for (TreeTypeParams& type : grown) {
            // Few, large leaves: far trees need few Gaussians.
            type = treeReducedType(type, 0.08f);
        }
    }
    const glm::vec4 up = treeUpRotation(glm::vec3(0.0f, 1.0f, 0.0f));
    std::vector<MossPatch> patches;
    heights.assign(types.size(), 0.0f);
    for (uint32_t tv = 0; tv < types.size() * variants; ++tv) {
        const uint32_t type = tv / variants;
        for (float season : seasons) {
            TreeFrame frame = defaultTreeFrame();
            frame.wind.w = 0.0f;
            frame.timeSeason.y = season;
            TreeInstance inst{};
            inst.posScale = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
            inst.rot = up;
            inst.ids = glm::uvec4(type, 1u + tv % variants, 0u, 0u);
            const TreeRecords records = generateTreesReference(grown, {inst}, frame);
            const TreeTypeParams& p = grown[type];
            float height = 0.0f;
            for (const SegmentRecord& s : records.segments) {
                height = std::max(height, std::max(s.fromPosRadius.y, s.toPosRadius.y));
            }
            for (const LeafRecord& l : records.leaves) {
                height = std::max(height, l.posScale.y + l.posScale.w);
            }
            heights[type] = std::max(heights[type], height);

            MossPatch patch;
            for (const LeafRecord& l : records.leaves) {
                const float length = l.posScale.w;
                if (length <= 0.0f) {
                    continue;
                }
                // A flat Gaussian over the blade: along +z, thin along its normal (+y); it stands for a cluster of
                // the full tree's leaves (or needles), so it is at least half as wide as long.
                const glm::vec3 center = glm::vec3(l.posScale) + 0.5f * length * tree::qforward(l.rot);
                const glm::vec3 albedo = tree::treeLeafSeasonAlbedo(p, l.ids.y, 1u, l.ids.w);
                patch.push_back(albedoGaussian(
                    center, l.rot,
                    glm::vec3(0.35f * length * std::max(p.leafCount.z, 0.6f), 0.04f * length, 0.4f * length), albedo,
                    0.9f));
            }
            for (const SegmentRecord& s : records.segments) {
                const glm::vec3 a(s.fromPosRadius), b(s.toPosRadius);
                const float radius = 0.5f * (s.fromPosRadius.w + s.toPosRadius.w);
                const float length = glm::length(b - a);
                if (radius < 1e-3f * height || length <= 0.0f) {
                    continue;
                }
                // Gaussians along the segment, about as long as the stem is thick.
                const int n = std::clamp(int(std::ceil(length / (2.0f * radius))), 1, 8);
                const glm::vec4 rot = tree::qnlerp(s.fromRot, s.toRot, 0.5f);
                const glm::vec3 bark(p.barkColor);
                for (int k = 0; k < n; ++k) {
                    const glm::vec3 c = glm::mix(a, b, (k + 0.5f) / n);
                    patch.push_back(albedoGaussian(c, rot, glm::vec3(radius, radius, 0.6f * length / n), bark, 0.95f));
                }
            }
            patches.push_back(std::move(patch));
        }
    }
    return patches;
}

TreeGaussianForest::TreeGaussianForest(VulkanContext& vulkanContext, TreeProxies& proxies,
                                       const GaussianSoABuffers& source, uint32_t poolOffset, uint32_t poolSlots,
                                       const std::vector<TreeTypeParams>& types, std::shared_ptr<CameraUboType> camera,
                                       std::shared_ptr<HostValues<TreeRenderSettings>> settings, BufferRef light,
                                       TreeGaussianForestConfig config) {
    if (config.seasons.empty() || config.seasons.size() > 4) {
        throw std::invalid_argument("TreeGaussianForest: needs 1 to 4 season samples");
    }
    const uint32_t typeCount = std::min<uint32_t>(uint32_t(types.size()), 2u);
    const std::vector<TreeTypeParams> used(types.begin(), types.begin() + typeCount);
    const uint32_t variants = std::max(config.variants, 1u);
    const std::vector<GaussianLods> patches =
        buildTreeGaussianLibraryLevels(used, config.seasons, heights, variants, config.budgets);

    struct Library {
        glm::vec4 seasons;
        glm::vec4 heights;
        glm::uvec4 counts;
        glm::uvec4 variants;
    } library{};
    for (uint32_t k = 0; k < 4; ++k) {
        library.seasons[k] = config.seasons[std::min<size_t>(k, config.seasons.size() - 1)];
    }
    // A copy is scaled as treeInstanceFor() scales the near tree of the same seed: the library's trees are its
    // trees at scale 1.
    library.heights = glm::vec4(used[0].shapeBase.z, used[std::min(1u, typeCount - 1)].shapeBase.z, 0.0f, 0.0f);
    library.variants = glm::uvec4(variants, config.seed, 0u, 0u);
    library.counts = glm::uvec4(uint32_t(config.seasons.size()), typeCount, config.trees, 0u);
    auto libraryBuffer = std::make_shared<BufferElementSinglePath<VulkanBuffer<Library>>>(vulkanContext, 1u);
    libraryBuffer->setName("TreeLibraryParams");
    libraryBuffer->getBuffer().memcopyFrom(std::vector<Library>{library});
    const uint32_t visibleCapacity = proxies.getConfig().capacity;
    // Read back by the tests, so host visible.
    instances = std::make_shared<BufferElement<VulkanBuffer<TreeLodInstance>>>(vulkanContext, visibleCapacity);
    instances->setName("TreeLibraryInstances");
    state = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanContext, TreeSizeCut::kStateSize);
    state->setName("TreeLibraryCut");
    // The bins are counted anew every frame; the cut-off before them carries over.
    state->setRecordToZeroRange(4 * sizeof(uint32_t), TREE_SIZE_BINS * sizeof(uint32_t));

    histogram =
        std::make_shared<GeneralComputation<uint32_t>>(vulkanContext, "shaders/tree/tree_library_histogram.comp.spv");
    histogram->setName("TreeLibraryHistogram");
    proxies.getVisible().connectTo(*histogram, 0);
    proxies.getCounter().connectTo(*histogram, 1);
    histogram->setInput(settings, 2);
    histogram->setInput(camera, 3);
    histogram->setInput(state, 4);
    proxies.getTileConfig().connectTo(*histogram, 5);
    histogram->addDependency(proxies.getElement());
    histogram->setGroupCountX((visibleCapacity + 255) / 256);
    histogram->setPushConstants({visibleCapacity});

    cutoff = std::make_shared<GeneralComputation<glm::uvec2>>(vulkanContext, "shaders/tree/tree_size_cut.comp.spv");
    cutoff->setName("TreeLibraryCutoff");
    cutoff->setInput(histogram, 0, 4);
    cutoff->setGroupCountX(1);
    // The copies the pool cannot hold even at the coarsest levels are left out from the next frame on (binding 1,
    // the level of detail's fitting, set below).
    cutoff->setPushConstants({glm::uvec2(config.trees, 1u)});

    convert =
        std::make_shared<GeneralComputation<uint32_t>>(vulkanContext, "shaders/tree/tree_library_instances.comp.spv");
    convert->setName("TreeLibraryInstances");
    proxies.getVisible().connectTo(*convert, 0);
    proxies.getCounter().connectTo(*convert, 1);
    convert->setInput(settings, 2);
    convert->setInput(libraryBuffer, 3);
    convert->setInput(instances, 4);
    convert->setInput(camera, 5);
    convert->setInput(cutoff, 6, 0);
    proxies.getTileConfig().connectTo(*convert, 7);
    convert->addDependency(proxies.getElement());
    convert->setGroupCountX((visibleCapacity + 63) / 64);
    convert->setPushConstants({visibleCapacity});

    lod = std::make_unique<TreeMossLod>(vulkanContext, source, patches, BufferRef{convert, 4}, visibleCapacity, camera,
                                        settings, light, config.boundaries, poolSlots, poolOffset, true);
    cutoff->setInput(lod->getFitState(), 1);
}

void TreeGaussianForest::initialize(uint32_t numberPaths) {
    for (uint32_t path = 0; path < numberPaths; ++path) {
        state->getBuffer(path).memcopyFrom(TreeSizeCut::initialState());
    }
    lod->resetFit();
}

TreeSizeCut TreeGaussianForest::readCut(uint32_t pathId) {
    std::vector<uint32_t> values(TreeSizeCut::kStateSize, 0u);
    state->getBuffer(pathId).memcopyTo(values);
    return TreeSizeCut::fromState(values);
}

std::vector<TreeLodInstance> TreeGaussianForest::readInstances(uint32_t pathId) {
    std::vector<TreeLodInstance> values(instances->getBuffer(pathId).getBufferMemSize() / sizeof(TreeLodInstance));
    instances->getBuffer(pathId).memcopyTo(values);
    return values;
}

} // namespace klartraum
