// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/tree/tree_gaussians.hpp"

namespace klartraum {

TreeGaussians::TreeGaussians(VulkanContext& vulkanContext, TreeGenerator& generator,
                             std::shared_ptr<CameraUboType> camera,
                             std::shared_ptr<HostValues<TreeRenderSettings>> settings,
                             std::vector<Gaussian3D> staticGaussiansIn, std::vector<TreeSceneVertex> surfels,
                             BufferRef light)
    : staticGaussians(std::move(staticGaussiansIn)),
      surfelCount(uint32_t(surfels.size())) {
    const TreeCapacity& capacity = generator.getCapacity();
    const uint32_t treeSlots = capacity.segments * kStemGaussians + capacity.leaves * kLeafGaussians;
    const uint32_t offset = uint32_t(staticGaussians.size());
    const uint32_t count = offset + surfelCount + treeSlots;

    if (surfels.empty()) {
        surfels.push_back(TreeSceneVertex{}); // the binding needs a buffer
    }
    auto surfelBuffer = std::make_shared<BufferElementSinglePath<VulkanBuffer<TreeSceneVertex>>>(
        vulkanContext, uint32_t(surfels.size()));
    surfelBuffer->setName("TreeSurfels");
    surfelBuffer->getBuffer().memcopyFrom(surfels);

    element = std::make_shared<GeneralComputation<glm::uvec4>>(vulkanContext, "shaders/tree/tree_gaussians.comp.spv");
    element->setName("TreeGaussians");
    generator.getSegments().connectTo(*element, 0);
    generator.getLeaves().connectTo(*element, 1);
    generator.getCounters().connectTo(*element, 2);
    generator.getParamsRef().connectTo(*element, 3);
    element->setInput(settings, 4);
    gaussians = addGaussianOutputs(vulkanContext, element, count, 5);
    element->setInput(camera, 12);
    const BufferRef lightRef = bufferOrPlaceholder(vulkanContext, light);
    element->setInput(lightRef.element, 13, lightRef.slot);
    element->setInput(surfelBuffer, 14);
    element->setGroupCountX((surfelCount + treeSlots + 255) / 256);
    element->setPushConstants({glm::uvec4(capacity.segments, capacity.leaves, offset, surfelCount)});
}

void TreeGaussians::initializeBuffers(uint32_t numberPaths) {
    const uint32_t count = gaussians.count;
    const uint32_t offset = uint32_t(staticGaussians.size());
    std::vector<glm::vec3> pos(offset), scale(offset);
    std::vector<glm::vec4> rot(offset), colAlpha(offset);
    std::vector<float> shR(15 * size_t(count), 0.0f), shG(15 * size_t(count), 0.0f), shB(15 * size_t(count), 0.0f);
    for (uint32_t i = 0; i < offset; ++i) {
        const Gaussian3D& g = staticGaussians[i];
        pos[i] = glm::vec3(g.position[0], g.position[1], g.position[2]);
        rot[i] = glm::vec4(g.rotation[0], g.rotation[1], g.rotation[2], g.rotation[3]);
        scale[i] = glm::vec3(g.scale[0], g.scale[1], g.scale[2]);
        colAlpha[i] = glm::vec4(g.color[0], g.color[1], g.color[2], g.alpha);
        for (int b = 0; b < 15; ++b) {
            shR[size_t(b) * count + i] = g.shR[b];
            shG[size_t(b) * count + i] = g.shG[b];
            shB[size_t(b) * count + i] = g.shB[b];
        }
    }
    auto upload = [&](const BufferRef& ref, const auto& data) {
        using T = typename std::decay_t<decltype(data)>::value_type;
        auto buffer = std::dynamic_pointer_cast<BufferElement<VulkanBuffer<T>>>(ref.buffer());
        for (uint32_t pathId = 0; pathId < numberPaths; ++pathId) {
            if (!data.empty()) {
                buffer->getBuffer(pathId).memcopyFrom(data);
            }
        }
    };
    upload(gaussians.pos, pos);
    upload(gaussians.rot, rot);
    upload(gaussians.scale, scale);
    upload(gaussians.colAlpha, colAlpha);
    upload(gaussians.shR, shR);
    upload(gaussians.shG, shG);
    upload(gaussians.shB, shB);
}

} // namespace klartraum
