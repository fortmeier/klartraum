// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/style_transfer.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace klartraum {

StyleTransfer::StyleTransfer(VulkanContext& vulkanContext, const std::string& modelPath,
                             std::shared_ptr<OffscreenTarget> color, std::shared_ptr<OffscreenTarget> output,
                             ComputeGraphElementPtr producer, std::shared_ptr<OffscreenTarget> classes,
                             std::shared_ptr<OffscreenTarget> depth) {
    const VkExtent2D frame = color->extent();
    network = vulkanContext.create<OnnxNetwork>(modelPath);
    network->setName("StyleTransfer");
    const std::vector<uint32_t> dimensions = network->getInputDimensions("input");
    const uint32_t channels = dimensions.at(1), height = dimensions.at(2), width = dimensions.at(3);
    if (width * 2 != frame.width || height * 2 != frame.height)
        throw std::runtime_error("StyleTransfer: the network's input is not half the frame's size");
    gbuffers = channels == 11;
    if (channels != 3 && !gbuffers)
        throw std::runtime_error("StyleTransfer: the network takes " + std::to_string(channels) +
                                 " input channels, not 3 or 11");
    if (gbuffers && (!classes || !depth))
        throw std::runtime_error("StyleTransfer: the network needs the classes and depth G-buffers");
    const Push push{frame.width, frame.height, width, height, channels, {0, 0, 0}};

    tensor = vulkanContext.create<TensorElement<float>>(std::vector<uint32_t>{1, channels, height, width});
    input = vulkanContext.create<GeneralComputation<Push>>("shaders/tree/style_transfer_input.comp.spv");
    input->setName("StyleTransferInput");
    input->setPushConstants({push});
    input->setGroupCount((width + 7) / 8, (height + 7) / 8, 1);
    // Without G-buffers, small images stand in for them: the shader reads them only for 11 channels.
    std::vector<std::shared_ptr<OffscreenTarget>> images = {color};
    if (gbuffers) {
        images.push_back(classes);
        images.push_back(depth);
    }
    for (size_t k = 0; k < 3; ++k) {
        if (k < images.size()) {
            input->setInput(images[k], int(k));
            input->setImageLayoutTransition(int(k), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                                            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                            VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        } else {
            auto standIn = std::make_shared<OffscreenTarget>(vulkanContext, VkExtent2D{1, 1}, 1);
            input->setInput(standIn, int(k));
            input->setImageLayoutTransition(int(k), VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                                            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                                            VK_ACCESS_SHADER_READ_BIT);
        }
    }
    input->setInput(tensor, 3);
    input->addDependency(producer);
    network->setInputTensor("input", input, 3);

    parameters = std::make_shared<HostValues<glm::vec4>>(vulkanContext, std::vector<glm::vec4>{glm::vec4(1.0f)});
    parameters->setName("StyleTransferBlend");
    toImage = vulkanContext.create<GeneralComputation<Push>>("shaders/tree/style_transfer_output.comp.spv");
    toImage->setName("StyleTransferOutput");
    toImage->setPushConstants({push});
    toImage->setGroupCount((frame.width + 7) / 8, (frame.height + 7) / 8, 1);
    toImage->setInput(network, 0, 0);
    toImage->setInput(input, 1, 0); // the colour, already in GENERAL
    toImage->setInput(output, 2);
    toImage->setImageLayoutTransition(2, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                                      VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                                      VK_ACCESS_SHADER_WRITE_BIT);
    toImage->setInput(parameters, 3);

    // Every image back in TRANSFER_SRC_OPTIMAL, the output last.
    ComputeGraphElementPtr previous = toImage;
    for (const auto& image : images) {
        auto back =
            std::make_shared<ImageViewSrcTransition>(VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        back->setInput(image, 0);
        back->addDependency(previous);
        previous = back;
    }
    auto ready =
        std::make_shared<ImageViewSrcTransition>(VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    ready->setName("StyleTransferReady");
    ready->setInput(toImage, 0, 2);
    ready->addDependency(previous);
    done = ready;
}

} // namespace klartraum
