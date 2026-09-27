#include "klartraum/gaussian_passes.hpp"

#include <string>

#include <glm/glm.hpp>

#include "klartraum/computegraph/bufferelement.hpp"
#include "klartraum/vulkan_buffer.hpp"

namespace klartraum {

namespace {

constexpr uint32_t kWorkgroup = 256;

// Per-path output buffers for `count` Gaussians, made inputs `first`..
// `first + 6` of `pass`; returns them as the pass's outputs.
template <typename Pass>
GaussianSoABuffers addOutputs(VulkanContext& vulkanContext, const std::shared_ptr<Pass>& pass, uint32_t count,
                              int first) {
    GaussianSoABuffers output;
    output.count = count;
    int slot = first;
    auto add = [&](auto element, const char* name) {
        element->setName(name);
        pass->setInput(element, slot);
        return BufferRef{pass, slot++};
    };
    output.pos = add(std::make_shared<BufferElement<VulkanBuffer<glm::vec3>>>(vulkanContext, count), "Pos3D");
    output.rot = add(std::make_shared<BufferElement<VulkanBuffer<glm::vec4>>>(vulkanContext, count), "Rot3D");
    output.scale = add(std::make_shared<BufferElement<VulkanBuffer<glm::vec3>>>(vulkanContext, count), "Scale3D");
    output.colAlpha =
        add(std::make_shared<BufferElement<VulkanBuffer<glm::vec4>>>(vulkanContext, count), "ColAlpha3D");
    output.shR = add(std::make_shared<BufferElement<VulkanBuffer<float>>>(vulkanContext, 15 * count), "ShR");
    output.shG = add(std::make_shared<BufferElement<VulkanBuffer<float>>>(vulkanContext, 15 * count), "ShG");
    output.shB = add(std::make_shared<BufferElement<VulkanBuffer<float>>>(vulkanContext, 15 * count), "ShB");
    return output;
}

void connectInputs(ComputeGraphElement& pass, const GaussianSoABuffers& source, int first) {
    int index = first;
    for (const BufferRef* ref : source.all()) {
        ref->connectTo(pass, index++);
    }
}

} // namespace

TransformBuffer makeTransformBuffer(VulkanContext& vulkanContext, const std::array<BufferRef, 7>& parameters) {
    auto pass = std::make_shared<TransformBufferPass>(vulkanContext);
    for (int i = 0; i < 7; ++i) {
        parameters[i].connectTo(*pass, i);
    }
    auto transform = std::make_shared<BufferElement<VulkanBuffer<float>>>(vulkanContext, TransformBufferPass::kSize);
    transform->setName("Transform");
    pass->setInput(transform, 7);
    pass->setGroupCount(1, 1, 1);
    return {pass, BufferRef{pass, 7}};
}

GaussianPass<GaussianTransformPass> transformGaussiansPass(VulkanContext& vulkanContext,
                                                          const GaussianSoABuffers& source,
                                                          const BufferRef& transform) {
    auto pass = std::make_shared<GaussianTransformPass>(vulkanContext);
    connectInputs(*pass, source, 0);
    transform.connectTo(*pass, 7);
    GaussianPass<GaussianTransformPass> result{pass, addOutputs(vulkanContext, pass, source.count, 8)};
    pass->setPushConstants({{source.count, 0u}});
    pass->setGroupCount((source.count + kWorkgroup - 1) / kWorkgroup, 1, 1);
    return result;
}

GaussianPass<GaussianMergePass> mergeGaussiansPass(VulkanContext& vulkanContext, const GaussianSoABuffers& a,
                                                   const GaussianSoABuffers& b) {
    auto pass = std::make_shared<GaussianMergePass>(vulkanContext);
    connectInputs(*pass, a, 0);
    connectInputs(*pass, b, 7);
    const uint32_t count = a.count + b.count;
    GaussianPass<GaussianMergePass> result{pass, addOutputs(vulkanContext, pass, count, 14)};
    pass->setPushConstants({{a.count, b.count}});
    pass->setGroupCount((count + kWorkgroup - 1) / kWorkgroup, 1, 1);
    return result;
}

} // namespace klartraum
