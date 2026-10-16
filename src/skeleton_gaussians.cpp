// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/computegraph/skeletongaussians.hpp"

#include <set>
#include <stdexcept>

#include "klartraum/computegraph/bufferelement.hpp"

namespace klartraum {

const std::vector<std::pair<uint32_t, uint32_t>>& characterSkeletonBones() {
    // Indices into CharacterController::kJointNames.
    static const std::vector<std::pair<uint32_t, uint32_t>> bones = {
        {0, 1},   {1, 2},   {2, 3},   {3, 4},                        // spine and head
        {1, 10},  {10, 11}, {11, 12}, {1, 13},  {13, 14}, {14, 15},  // arms
        {0, 16},  {16, 20}, {0, 20},                                 // pelvis
        {16, 17}, {17, 18}, {18, 19}, {20, 21}, {21, 22}, {22, 23}}; // legs
    return bones;
}

SkeletonGaussiansResult createSkeletonGaussians(VulkanContext& vulkanContext, const BufferRef& joints,
                                                const std::vector<std::pair<uint32_t, uint32_t>>& bones,
                                                const BufferRef& params, uint32_t samplesPerBone) {
    if (bones.empty() || samplesPerBone == 0) {
        throw std::invalid_argument("createSkeletonGaussians: need bones and at least one sample per bone");
    }
    std::vector<uint32_t> indices;
    std::set<uint32_t> used;
    for (const auto& [a, b] : bones) {
        indices.push_back(a);
        indices.push_back(b);
        used.insert(a);
        used.insert(b);
    }
    // The joints that get a sphere follow the bone pairs in the same buffer.
    indices.insert(indices.end(), used.begin(), used.end());

    auto element = std::make_shared<SkeletonGaussians>(vulkanContext);
    joints.connectTo(*element, 0);
    auto indexBuffer = std::make_shared<BufferElementSinglePath<VulkanBuffer<uint32_t>>>(
        vulkanContext, uint32_t(indices.size()), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    indexBuffer->setName("SkeletonIndices");
    indexBuffer->getBuffer().memcopyFrom(indices);
    element->setInput(indexBuffer, 1);
    params.connectTo(*element, 2);

    const uint32_t boneCount = uint32_t(bones.size());
    const uint32_t jointCount = uint32_t(used.size());
    const uint32_t total = boneCount * samplesPerBone + jointCount;
    SkeletonGaussiansResult result{element, addGaussianOutputs(vulkanContext, element, total, 3)};
    element->setPushConstants({{boneCount, samplesPerBone, jointCount, total}});
    element->setGroupCount((total + 63) / 64, 1, 1);
    return result;
}

} // namespace klartraum
