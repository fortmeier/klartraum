// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/tree/tree_library.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

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

} // namespace

std::vector<MossPatch> buildTreeGaussianLibrary(const std::vector<TreeTypeParams>& types,
                                                const std::vector<float>& seasons, std::vector<float>& heights,
                                                bool reduced) {
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
    for (uint32_t type = 0; type < types.size(); ++type) {
        for (float season : seasons) {
            TreeFrame frame = defaultTreeFrame();
            frame.wind.w = 0.0f;
            frame.timeSeason.y = season;
            TreeInstance inst{};
            inst.posScale = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
            inst.rot = up;
            inst.ids = glm::uvec4(type, 1u, 0u, 0u);
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
    const std::vector<MossPatch> patches = buildTreeGaussianLibrary(used, config.seasons, heights);

    struct Library {
        glm::vec4 seasons;
        glm::vec4 heights;
        glm::uvec4 counts;
    } library{};
    for (uint32_t k = 0; k < 4; ++k) {
        library.seasons[k] = config.seasons[std::min<size_t>(k, config.seasons.size() - 1)];
    }
    library.heights = glm::vec4(heights[0], heights[std::min(1u, typeCount - 1)], 0.0f, 0.0f);
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

    cutoff = std::make_shared<GeneralComputation<uint32_t>>(vulkanContext, "shaders/tree/tree_size_cut.comp.spv");
    cutoff->setName("TreeLibraryCutoff");
    cutoff->setInput(histogram, 0, 4);
    cutoff->setGroupCountX(1);
    cutoff->setPushConstants({config.trees});

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
                                        settings, light, config.boundaries, poolSlots, poolOffset);
}

void TreeGaussianForest::initialize(uint32_t numberPaths) {
    for (uint32_t path = 0; path < numberPaths; ++path) {
        state->getBuffer(path).memcopyFrom(TreeSizeCut::initialState());
    }
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
