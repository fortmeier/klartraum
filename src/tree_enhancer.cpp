// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/tree/tree_enhancer.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace klartraum {

TreeEnhancer::TreeEnhancer(VulkanContext& vulkanContext, const std::string& modelPath,
                           std::shared_ptr<OffscreenTarget> color, std::shared_ptr<OffscreenTarget> featureA,
                           std::shared_ptr<OffscreenTarget> featureB, std::shared_ptr<OffscreenTarget> output,
                           ComputeGraphElementPtr producer) {
    const VkExtent2D extent = color->extent();
    const Push push{extent.width, extent.height};
    network = vulkanContext.create<OnnxNetwork>(modelPath);
    network->setName("TreeEnhancer");
    const uint32_t channels = network->getInputDimensions("input").at(1);
    if (channels != 8 && channels != 9) {
        throw std::runtime_error("TreeEnhancer: the network takes " + std::to_string(channels) +
                                 " input channels, not 8 or 9");
    }
    tensor =
        vulkanContext.create<TensorElement<float>>(std::vector<uint32_t>{1, channels, extent.height, extent.width});
    input = vulkanContext.create<GeneralComputation<InputPush>>("shaders/tree/tree_enhancer_input.comp.spv");
    input->setName("TreeEnhancerInput");
    input->setPushConstants({InputPush{extent.width, extent.height, channels, 0}});
    input->setGroupCount((extent.width + 7) / 8, (extent.height + 7) / 8, 1);
    const std::shared_ptr<OffscreenTarget> images[3] = {color, featureA, featureB};
    for (int k = 0; k < 3; ++k) {
        input->setInput(images[k], k);
        input->setImageLayoutTransition(k, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                                        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                        VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    }
    input->setInput(tensor, 3);
    input->addDependency(producer);

    network->setInputTensor("input", input, 3);

    toImage = vulkanContext.create<GeneralComputation<Push>>("shaders/onnx/tensor_to_image.comp.spv");
    toImage->setName("TreeEnhancerOutput");
    toImage->setPushConstants({push});
    toImage->setGroupCount((extent.width + 7) / 8, (extent.height + 7) / 8, 1);
    toImage->setInput(network, 0, 0);
    toImage->setInput(output, 1);
    toImage->setImageLayoutTransition(1, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                                      VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                                      VK_ACCESS_SHADER_WRITE_BIT);

    // Every image back in TRANSFER_SRC_OPTIMAL, the output last.
    ComputeGraphElementPtr previous = toImage;
    for (int k = 0; k < 3; ++k) {
        auto back =
            std::make_shared<ImageViewSrcTransition>(VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        back->setInput(images[k], 0);
        back->addDependency(previous);
        previous = back;
    }
    auto ready =
        std::make_shared<ImageViewSrcTransition>(VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    ready->setName("TreeEnhancerReady");
    ready->setInput(toImage, 0, 1);
    ready->addDependency(previous);
    done = ready;
}

} // namespace klartraum
