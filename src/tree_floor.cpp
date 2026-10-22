// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/tree/tree_floor.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace klartraum {

float mossPatchRadius(const std::vector<MossPatch>& patches) {
    float radius = 0.0f;
    for (const MossPatch& patch : patches) {
        for (const Gaussian3D& g : patch) {
            radius = std::max(radius, std::sqrt(g.position[0] * g.position[0] + g.position[2] * g.position[2]));
        }
    }
    return radius;
}

TreeFloorSetup makeTreeFloor(const BonsaiScene& scene, uint32_t patchCount, float patchRadius, float scale,
                             TreeFloorConfig config) {
    if (patchCount == 0 || patchRadius <= 0.0f || config.radiusMeters <= 0.0f) {
        throw std::invalid_argument("makeTreeFloor: needs patches of a size and a radius");
    }
    const float mpu = scene.config.scale.metersPerUnit();
    TreeFloorSetup f{};
    f.terrain = scene.terrain;
    const float cell = config.spacing * patchRadius * scale;
    const float radius = config.radiusMeters / mpu;
    f.grid = glm::vec4(cell, radius, scale, config.fadeFrom);
    f.counts = glm::uvec4(2u * uint32_t(std::ceil(radius / cell)) + 2u, patchCount, config.seed, 0u);
    return f;
}

glm::ivec2 treeFloorGridOrigin(const TreeFloorSetup& floor, glm::vec3 eye) {
    const int half = int(floor.counts.x / 2u);
    return {int(std::floor(eye.x / floor.grid.x)) - half + 1, int(std::floor(eye.z / floor.grid.x)) - half + 1};
}

std::vector<TreeLodInstance> treeFloorAround(const TreeFloorSetup& floor, glm::vec3 eye) {
    const glm::ivec2 origin = treeFloorGridOrigin(floor, eye);
    std::vector<TreeLodInstance> copies;
    copies.reserve(size_t(floor.counts.x) * floor.counts.x);
    for (uint32_t j = 0; j < floor.counts.x; ++j) {
        for (uint32_t i = 0; i < floor.counts.x; ++i) {
            copies.push_back(tree::treeFloorCell(floor, origin.x + int(i), origin.y + int(j), eye));
        }
    }
    return copies;
}

TreeForestFloor::TreeForestFloor(VulkanContext& vulkanContext, const TreeFloorSetup& floor,
                                 const GaussianSoABuffers& source, const std::vector<MossPatch>& patches,
                                 std::shared_ptr<CameraUboType> camera,
                                 std::shared_ptr<HostValues<TreeRenderSettings>> settings, BufferRef light,
                                 glm::vec4 boundaries, uint32_t poolSlots, uint32_t poolOffset)
    : setup(floor) {
    const uint32_t n = floor.counts.x;
    auto setupBuffer = std::make_shared<BufferElementSinglePath<VulkanBuffer<TreeFloorSetup>>>(vulkanContext, 1u);
    setupBuffer->setName("TreeFloorSetup");
    setupBuffer->getBuffer().memcopyFrom(std::vector<TreeFloorSetup>{floor});
    instances = std::make_shared<BufferElement<VulkanBuffer<TreeLodInstance>>>(vulkanContext, n * n);
    instances->setName("TreeFloorInstances");
    frame = std::make_shared<HostValues<glm::vec4>>(vulkanContext,
                                                    std::vector<glm::vec4>{glm::vec4(1.0f, 0.0f, 0.0f, 0.0f)});
    frame->setName("TreeFloorFrame");

    place = std::make_shared<GeneralComputation<uint32_t>>(vulkanContext, "shaders/tree/tree_floor_place.comp.spv");
    place->setName("TreeFloorPlace");
    place->setInput(setupBuffer, 0);
    place->setInput(camera, 1);
    place->setInput(settings, 2);
    place->setInput(instances, 3);
    place->setInput(frame, 4);
    const uint32_t groups = (n + 15u) / 16u;
    place->setGroupCountX(groups);
    place->setGroupCountY(groups);
    place->setPushConstants({0u});

    lod = std::make_unique<TreeMossLod>(vulkanContext, source, patches, BufferRef{place, 3}, n * n, camera, settings,
                                        light, boundaries, poolSlots, poolOffset);
}

void TreeForestFloor::setView(glm::vec3 eye) {
    const float above = eye.y - tree::treeTerrainHeight(setup.terrain, eye.x, eye.z);
    frame->set(0, glm::vec4(above <= setup.grid.y ? 1.0f : 0.0f, 0.0f, 0.0f, 0.0f));
}

std::vector<TreeLodInstance> TreeForestFloor::readInstances(uint32_t pathId) {
    std::vector<TreeLodInstance> values(size_t(setup.counts.x) * setup.counts.x);
    instances->getBuffer(pathId).memcopyTo(values);
    return values;
}

} // namespace klartraum
