// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/raster/differentiable_shading.hpp"

#include <stdexcept>
#include <string>
#include <vector>

#include "klartraum/computegraph/bufferelement.hpp"
#include "klartraum/computegraph/generalcomputation.hpp"

namespace klartraum::raster {

namespace {

struct ShadingPushConstants {
    uint32_t width, height, textureWidth, textureHeight;
};

struct ImageLossPushConstants {
    uint32_t width, height;
    float maskWeight;
};

struct ViewColorLossPushConstants {
    uint32_t width, height;
    float maskWeight, viewColorWeight;
};

struct RegularizerPushConstants {
    uint32_t vertexCount;
    float weight;
};

struct ViewCountPushConstants {
    uint32_t viewCount;
};

ComputeGraphElementPtr shading(VulkanContext& vulkanContext, const char* shader, uint32_t width, uint32_t height,
                               uint32_t textureWidth, uint32_t textureHeight) {
    if (width == 0 || height == 0 || textureWidth == 0 || textureHeight == 0)
        throw std::invalid_argument("shading: image and texture sizes must be positive");
    auto operation = vulkanContext.create<GeneralComputation<ShadingPushConstants>>(shader);
    operation->setPushConstants({{width, height, textureWidth, textureHeight}});
    operation->setGroupCount((width + 7) / 8, (height + 7) / 8, 1);
    return operation;
}

// Fills its input buffers with zeros.
class ClearBuffers : public ComputeGraphElement {
public:
    const char* getType() const override { return "ClearBuffers"; }
    void checkInput(ComputeGraphElementPtr input, int index) override {
        if (dynamic_cast<BufferElementInterface*>(input.get()) == nullptr)
            throw std::runtime_error("clearBuffers: every input must be a buffer or tensor");
    }
    void _record(VkCommandBuffer commandBuffer, uint32_t pathId) override {
        for (const auto& [index, input] : getInputs()) {
            auto* buffer = dynamic_cast<BufferElementInterface*>(getInputElement(index).get());
            vkCmdFillBuffer(commandBuffer, buffer->getVkBuffer(pathId), 0, VK_WHOLE_SIZE, 0);
        }
    }
};

} // namespace

ComputeGraphElementPtr shade(VulkanContext& vulkanContext, uint32_t width, uint32_t height, uint32_t textureWidth,
                             uint32_t textureHeight) {
    return shading(vulkanContext, "shaders/raster/shade.comp.spv", width, height, textureWidth, textureHeight);
}

ComputeGraphElementPtr shadeBackward(VulkanContext& vulkanContext, uint32_t width, uint32_t height,
                                     uint32_t textureWidth, uint32_t textureHeight) {
    return shading(vulkanContext, "shaders/raster/shade_backward.comp.spv", width, height, textureWidth, textureHeight);
}

ComputeGraphElementPtr shadeColorBackward(VulkanContext& vulkanContext, uint32_t width, uint32_t height,
                                          uint32_t textureWidth, uint32_t textureHeight) {
    return shading(vulkanContext, "shaders/raster/shade_color_backward.comp.spv", width, height, textureWidth,
                   textureHeight);
}

ComputeGraphElementPtr antialias(VulkanContext& vulkanContext, uint32_t width, uint32_t height, uint32_t textureWidth,
                                 uint32_t textureHeight) {
    return shading(vulkanContext, "shaders/raster/antialias.comp.spv", width, height, textureWidth, textureHeight);
}

ComputeGraphElementPtr antialiasBackward(VulkanContext& vulkanContext, uint32_t width, uint32_t height,
                                         uint32_t textureWidth, uint32_t textureHeight) {
    return shading(vulkanContext, "shaders/raster/antialias_backward.comp.spv", width, height, textureWidth,
                   textureHeight);
}

ComputeGraphElementPtr imageLoss(VulkanContext& vulkanContext, uint32_t width, uint32_t height, float maskWeight) {
    if (width == 0 || height == 0)
        throw std::invalid_argument("imageLoss: the image size must be positive");
    auto operation =
        vulkanContext.create<GeneralComputation<ImageLossPushConstants>>("shaders/raster/image_loss.comp.spv");
    operation->setPushConstants({{width, height, maskWeight}});
    operation->setGroupCountX((width * height + 255) / 256);
    return operation;
}

ComputeGraphElementPtr imageLossWithViewColors(VulkanContext& vulkanContext, uint32_t width, uint32_t height,
                                               float maskWeight, float viewColorWeight) {
    if (width == 0 || height == 0)
        throw std::invalid_argument("imageLossWithViewColors: the image size must be positive");
    auto operation = vulkanContext.create<GeneralComputation<ViewColorLossPushConstants>>(
        "shaders/raster/image_loss_view_colors.comp.spv");
    operation->setPushConstants({{width, height, maskWeight, viewColorWeight}});
    operation->setGroupCountX((width * height + 255) / 256);
    return operation;
}

ComputeGraphElementPtr laplacianRegularizer(VulkanContext& vulkanContext, uint32_t vertexCount, float weight) {
    if (vertexCount == 0)
        throw std::invalid_argument("laplacianRegularizer: the vertex count must be positive");
    auto operation = vulkanContext.create<GeneralComputation<RegularizerPushConstants>>(
        std::vector<std::string>{"shaders/raster/laplacian.comp.spv", "shaders/raster/laplacian_gradient.comp.spv"});
    operation->setPushConstants({{vertexCount, weight}});
    // The first pass writes every vertex's Laplacian before the second reads it.
    operation->addScratchBufferElement(
        std::make_shared<BufferElement<VulkanBuffer<float>>>(vulkanContext, size_t(vertexCount) * 4), false);
    operation->setGroupCountX((vertexCount + 255) / 256);
    return operation;
}

ComputeGraphElementPtr clearBuffers(VulkanContext& vulkanContext) { return std::make_shared<ClearBuffers>(); }

ComputeGraphElementPtr cycleView(VulkanContext& vulkanContext, uint32_t viewCount) {
    if (viewCount == 0)
        throw std::invalid_argument("cycleView: the view count must be positive");
    auto operation =
        vulkanContext.create<GeneralComputation<ViewCountPushConstants>>("shaders/raster/cycle_view.comp.spv");
    operation->setPushConstants({{viewCount}});
    operation->setGroupCountX(1);
    return operation;
}

} // namespace klartraum::raster
