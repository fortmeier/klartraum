// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/field_style.hpp"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "klartraum/style_transfer.hpp"

namespace klartraum {

namespace {

constexpr uint32_t kMagic = 0x444C464B; // "KFLD"
constexpr uint32_t kInputChannels = 41;

// The field file: a header of 12 uint32 (magic, version 2, channels per grid corner, the coarse grid's size xyz, the
// fine grid's, the paper tile's planes, height and width) and the box (low xyz, high xyz, floats), then the coarse
// grid, the fine grid and the paper as floats.
struct FieldFile {
    uint32_t header[12];
    float box[6];
    std::vector<float> data;
};

FieldFile readField(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    FieldFile field;
    file.read(reinterpret_cast<char*>(field.header), sizeof(field.header));
    file.read(reinterpret_cast<char*>(field.box), sizeof(field.box));
    if (!file || field.header[0] != kMagic || field.header[1] != 2 || field.header[2] != 8 || field.header[9] != 4)
        throw std::runtime_error("FieldStyle: " + path + " is no field of version 2");
    const auto product = [&](int first) {
        return size_t(field.header[first]) * field.header[first + 1] * field.header[first + 2];
    };
    field.data.resize(8 * product(3) + 8 * product(6) + product(9));
    file.read(reinterpret_cast<char*>(field.data.data()), std::streamsize(field.data.size() * sizeof(float)));
    if (!file)
        throw std::runtime_error("FieldStyle: " + path + " is truncated");
    return field;
}

// A constant tensor that copies its values into its buffer when the graph sets it up.
class ConstantData : public TensorElementSinglePath<float> {
public:
    ConstantData(VulkanContext& vulkanContext, std::vector<float> data)
        : TensorElementSinglePath<float>(vulkanContext, std::vector<uint32_t>{uint32_t(data.size())}),
          values(std::move(data)) {}

    void _setup(VulkanContext& vulkanContext, uint32_t numberPaths, VkQueue setupQueue) override {
        TensorElementSinglePath<float>::_setup(vulkanContext, numberPaths, setupQueue);
        getDataBuffer(0).memcopyFrom(values);
    }

private:
    // The values to copy; kept until the graph is set up.
    std::vector<float> values;
};

} // namespace

FieldStyle::FieldStyle(VulkanContext& vulkanContext, const std::string& modelPath, const std::string& fieldPath,
                       std::shared_ptr<OffscreenTarget> color, std::shared_ptr<OffscreenTarget> classes,
                       std::shared_ptr<OffscreenTarget> coverage, std::shared_ptr<OffscreenTarget> depth,
                       std::shared_ptr<OffscreenTarget> output, ComputeGraphElementPtr producer) {
    const VkExtent2D frame = color->extent();
    FieldFile field = readField(fieldPath);
    const uint32_t* h = field.header;
    low = glm::vec3(field.box[0], field.box[1], field.box[2]);
    high = glm::vec3(field.box[3], field.box[4], field.box[5]);
    network = vulkanContext.create<OnnxNetwork>(modelPath);
    network->setName("FieldStyle");
    const std::vector<uint32_t> dimensions = network->getInputDimensions("input");
    if (dimensions.at(1) != kInputChannels || dimensions.at(2) != frame.height || dimensions.at(3) != frame.width)
        throw std::runtime_error("FieldStyle: the network does not take 41 planes at the frame's size");

    const uint32_t fineOffset = 8 * h[3] * h[4] * h[5], paperOffset = fineOffset + 8 * h[6] * h[7] * h[8];
    const Push push{frame.width, frame.height, h[3],        h[4],  h[5],  h[6],  h[7],   h[8],   h[11],
                    h[10],       fineOffset,   paperOffset, low.x, low.y, low.z, high.x, high.y, high.z};
    fieldData = std::make_shared<ConstantData>(vulkanContext, std::move(field.data));
    frameValues = std::make_shared<HostValues<glm::vec4>>(vulkanContext, 8);
    frameValues->setName("FieldStyleFrame");

    tensor =
        vulkanContext.create<TensorElement<float>>(std::vector<uint32_t>{1, kInputChannels, frame.height, frame.width});
    input = vulkanContext.create<GeneralComputation<Push>>("shaders/tree/style_field_input.comp.spv");
    input->setName("FieldStyleInput");
    input->setPushConstants({push});
    input->setGroupCount((frame.width + 7) / 8, (frame.height + 7) / 8, 1);
    const std::shared_ptr<OffscreenTarget> images[] = {color, classes, depth, coverage};
    const int slots[] = {0, 1, 2, 6};
    for (int k = 0; k < 4; ++k) {
        input->setInput(images[k], slots[k]);
        input->setImageLayoutTransition(slots[k], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                                        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                        VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    }
    input->setInput(tensor, 3);
    input->setInput(frameValues, 4);
    input->setInput(fieldData, 5);
    input->addDependency(producer);
    network->setInputTensor("input", input, 3);

    // The output pass of StyleTransfer, at the frame's size.
    blendValues = std::make_shared<HostValues<glm::vec4>>(vulkanContext, std::vector<glm::vec4>{glm::vec4(1.0f)});
    blendValues->setName("FieldStyleBlend");
    const StyleTransfer::Push outputPush{frame.width, frame.height, frame.width, frame.height, 3, 1, {0, 0}};
    auto toImage =
        vulkanContext.create<GeneralComputation<StyleTransfer::Push>>("shaders/tree/style_transfer_output.comp.spv");
    toImage->setName("FieldStyleOutput");
    toImage->setPushConstants({outputPush});
    toImage->setGroupCount((frame.width + 7) / 8, (frame.height + 7) / 8, 1);
    toImage->setInput(network, 0, 0);
    toImage->setInput(input, 1, 0); // the colour, already in GENERAL
    toImage->setInput(output, 2);
    toImage->setImageLayoutTransition(2, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                                      VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                                      VK_ACCESS_SHADER_WRITE_BIT);
    toImage->setInput(blendValues, 3);

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
    ready->setName("FieldStyleReady");
    ready->setInput(toImage, 0, 2);
    ready->addDependency(previous);
    done = ready;
}

void FieldStyle::setFrame(const glm::mat4& camera, glm::vec3 eye, float near, float far,
                          const std::array<float, 7>& conditions) {
    const glm::mat4 inverse = glm::inverse(camera);
    for (int k = 0; k < 4; ++k)
        frameValues->set(uint32_t(k), inverse[k]);
    frameValues->set(4, glm::vec4(eye, 0.0f));
    frameValues->set(5, glm::vec4(near, far, 0.0f, 0.0f));
    frameValues->set(6, glm::vec4(conditions[0], conditions[1], conditions[2], conditions[3]));
    frameValues->set(7, glm::vec4(conditions[4], conditions[5], conditions[6], 0.0f));
}

std::pair<float, float> fieldDepthWindow(glm::vec3 eye, glm::vec3 low, glm::vec3 high) {
    const float nearest = glm::length(eye - glm::clamp(eye, low, high));
    float farthest = 0.0f;
    for (int corner = 0; corner < 8; ++corner) {
        const glm::vec3 c((corner & 1) ? high.x : low.x, (corner & 2) ? high.y : low.y, (corner & 4) ? high.z : low.z);
        farthest = std::max(farthest, glm::length(c - eye));
    }
    return {std::max(nearest, 0.05f), farthest};
}

} // namespace klartraum
