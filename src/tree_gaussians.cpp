// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/tree/tree_gaussians.hpp"

namespace klartraum {

TreeGaussians::TreeGaussians(VulkanContext& vulkanContext, TreeGenerator& generator,
                             std::shared_ptr<CameraUboType> camera,
                             std::shared_ptr<HostValues<TreeRenderSettings>> settings) {
    const TreeCapacity& capacity = generator.getCapacity();
    const uint32_t count = capacity.segments * kStemGaussians + capacity.leaves * kLeafGaussians;

    element = std::make_shared<GeneralComputation<glm::uvec2>>(vulkanContext, "shaders/tree/tree_gaussians.comp.spv");
    element->setName("TreeGaussians");
    generator.getSegments().connectTo(*element, 0);
    generator.getLeaves().connectTo(*element, 1);
    generator.getCounters().connectTo(*element, 2);
    generator.getParamsRef().connectTo(*element, 3);
    element->setInput(settings, 4);
    gaussians = addGaussianOutputs(vulkanContext, element, count, 5);
    element->setInput(camera, 12);
    element->setGroupCountX((count + 255) / 256);
    element->setPushConstants({glm::uvec2(capacity.segments, capacity.leaves)});
}

void TreeGaussians::zeroHigherBands(uint32_t numberPaths) {
    for (const BufferRef* sh : {&gaussians.shR, &gaussians.shG, &gaussians.shB}) {
        auto buffer = std::dynamic_pointer_cast<BufferElement<VulkanBuffer<float>>>(sh->buffer());
        for (uint32_t pathId = 0; pathId < numberPaths; ++pathId) {
            buffer->getBuffer(pathId).zero();
        }
    }
}

} // namespace klartraum
