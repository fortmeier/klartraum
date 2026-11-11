// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/neural/vgg_loss.hpp"

#include <filesystem>
#include <fstream>
#include <stdexcept>

#include "klartraum/layers/layers.hpp"
#include "klartraum/layers/training.hpp"

namespace klartraum::neural {

namespace {

// ImageNet's channel means and standard deviations (torchvision's normalization).
const std::vector<float> kMean{0.485f, 0.456f, 0.406f}, kStd{0.229f, 0.224f, 0.225f};
// The convolutions after which the loss takes the features (relu1_2, relu2_2, relu3_3, relu4_3) and their weights.
const uint32_t kTaps[4] = {1, 3, 6, 9};
const float kTapWeights[4] = {1.0f / 32.0f, 1.0f / 16.0f, 1.0f / 8.0f, 1.0f / 4.0f};

std::vector<float> readFloats(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
        throw std::runtime_error("VggLoss: cannot read " + path.string() + " (scripts/neural/export_vgg16.py)");
    std::vector<float> data(size_t(file.tellg()) / sizeof(float));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(data.data()), std::streamsize(data.size() * sizeof(float)));
    return data;
}

} // namespace

// The activations of one image batch through VGG: the normalized input, every convolution's output, its ReLU and,
// where one follows, its max-pool with the taken elements.
struct VggLoss::Path {
    std::shared_ptr<TensorElement<float>> normalized;
    std::vector<std::shared_ptr<TensorElement<float>>> convolved, activated, pooled, taken;
    std::vector<std::vector<uint32_t>> shapes; ///< of each convolution's output
};

VggLoss::VggLoss(VulkanContext& vulkanContext, const std::string& directory, uint32_t batch, uint32_t height,
                 uint32_t width, float weight, std::shared_ptr<TensorElement<float>> images,
                 std::shared_ptr<TensorElement<float>> targets, std::shared_ptr<TensorElement<float>> imageGradient)
    : vulkanContext(vulkanContext),
      directory(directory),
      batch(batch),
      height(height),
      width(width) {
    std::ifstream layers(std::filesystem::path(directory) / "layers.txt");
    for (uint32_t in, out, pool; layers >> in >> out >> pool;) {
        inputChannels.push_back(in);
        outputChannels.push_back(out);
        pooled.push_back(pool != 0);
        weights.push_back(tensor({out, in, 3, 3}));
        biases.push_back(tensor({out}));
    }
    if (outputChannels.size() != 10)
        throw std::runtime_error("VggLoss: " + directory + " has no VGG-16 up to relu4_3");
    if (height % 8 != 0 || width % 8 != 0)
        throw std::invalid_argument("VggLoss: the images must halve exactly three times");

    Path image, target;
    forward(image, images);
    forward(target, targets);

    // The deepest feature's distance first; going back through VGG, each shallower feature's distance adds its
    // gradient where the gradient from deeper layers arrives.
    std::vector<std::shared_ptr<TensorElement<float>>> gradients(10); // of each ReLU's output
    for (int tap = 3, k = 9; k >= 0; --k) {
        const auto& shape = image.shapes[size_t(k)];
        if (tap >= 0 && uint32_t(k) == kTaps[tap]) {
            const uint32_t count = uint32_t(image.activated[size_t(k)]->getDataElementCount());
            const bool deepest = k == 9;
            if (deepest)
                gradients[size_t(k)] = tensor(shape);
            auto shares = tensor(shape);
            auto distance = training::l1Distance(vulkanContext, count, weight * kTapWeights[tap], !deepest);
            distance->setInput(image.activated[size_t(k)], 0);
            distance->setInput(target.activated[size_t(k)], 1);
            distance->setInput(gradients[size_t(k)], 2);
            distance->setInput(shares, 3);
            sequence.push_back(distance);
            auto total = training::sum(vulkanContext, count);
            totals.push_back(tensor({1}));
            total->setInput(shares, 0);
            total->setInput(totals.back(), 1);
            sequence.push_back(total);
            --tap;
        }
        auto preGradient = tensor(shape);
        auto relu = training::activationBackward(vulkanContext, training::Activation::Relu, shape);
        relu->setInput(image.convolved[size_t(k)], 0);
        relu->setInput(gradients[size_t(k)], 1);
        relu->setInput(preGradient, 2);
        sequence.push_back(relu);
        // The convolution's input (stride 1, padding 1: of its output's size): the normalized image, the previous
        // ReLU's output, or its max-pool.
        const bool fromPool = k > 0 && pooled[size_t(k) - 1];
        const std::vector<uint32_t> inputShape{batch, inputChannels[size_t(k)], shape[2], shape[3]};
        auto inputGradient = tensor(inputShape);
        layers::ConvAttributes attributes;
        attributes.kernelShape = {3, 3};
        attributes.pads = {1, 1, 1, 1};
        auto conv = training::convBackwardInput(vulkanContext, attributes, inputShape,
                                                {outputChannels[size_t(k)], inputChannels[size_t(k)], 3, 3}, shape);
        conv->setInput(preGradient, 0);
        conv->setInput(weights[size_t(k)], 1);
        conv->setInput(inputGradient, 2);
        sequence.push_back(conv);
        if (k == 0) {
            std::vector<float> scale(3), shift(3, 0.0f);
            for (size_t c = 0; c < 3; ++c)
                scale[c] = 1.0f / kStd[c];
            auto affine = training::channelAffine(vulkanContext, inputShape, scale, shift, true, true);
            affine->setInput(inputGradient, 0);
            affine->setInput(imageGradient, 1);
            sequence.push_back(affine);
        } else if (fromPool) {
            const auto& previous = image.shapes[size_t(k) - 1];
            gradients[size_t(k) - 1] = tensor(previous);
            auto pool = training::maxPoolBackward(vulkanContext, previous);
            pool->setInput(inputGradient, 0);
            pool->setInput(image.taken[size_t(k) - 1], 1);
            pool->setInput(gradients[size_t(k) - 1], 2);
            sequence.push_back(pool);
        } else {
            gradients[size_t(k) - 1] = inputGradient;
        }
    }
    for (size_t i = 1; i < sequence.size(); ++i)
        sequence[i]->addDependency(sequence[i - 1]);
}

std::shared_ptr<TensorElement<float>> VggLoss::tensor(const std::vector<uint32_t>& shape) {
    return vulkanContext.create<TensorElement<float>>(shape);
}

void VggLoss::forward(Path& path, std::shared_ptr<TensorElement<float>> images) {
    std::vector<float> scale(3), shift(3);
    for (size_t c = 0; c < 3; ++c) {
        scale[c] = 1.0f / kStd[c];
        shift[c] = -kMean[c] / kStd[c];
    }
    path.normalized = tensor({batch, 3, height, width});
    auto normalize = training::channelAffine(vulkanContext, {batch, 3, height, width}, scale, shift);
    normalize->setInput(images, 0);
    normalize->setInput(path.normalized, 1);
    sequence.push_back(normalize);
    auto x = path.normalized;
    uint32_t h = height, w = width;
    for (size_t k = 0; k < outputChannels.size(); ++k) {
        const std::vector<uint32_t> shape{batch, outputChannels[k], h, w};
        path.shapes.push_back(shape);
        path.convolved.push_back(tensor(shape));
        path.activated.push_back(tensor(shape));
        layers::ConvAttributes attributes;
        attributes.kernelShape = {3, 3};
        attributes.pads = {1, 1, 1, 1};
        auto conv = layers::conv(vulkanContext, attributes, {batch, inputChannels[k], h, w},
                                 {outputChannels[k], inputChannels[k], 3, 3}, {outputChannels[k]}, shape);
        conv->setInput(x, 0);
        conv->setInput(weights[k], 1);
        conv->setInput(biases[k], 2);
        conv->setInput(path.convolved[k], 3);
        sequence.push_back(conv);
        auto relu = layers::relu(vulkanContext, shape);
        relu->setInput(path.convolved[k], 0);
        relu->setInput(path.activated[k], 1);
        sequence.push_back(relu);
        x = path.activated[k];
        if (pooled[k]) {
            h /= 2;
            w /= 2;
            path.pooled.push_back(tensor({batch, outputChannels[k], h, w}));
            path.taken.push_back(tensor({batch, outputChannels[k], h, w}));
            auto pool = training::maxPool(vulkanContext, shape);
            pool->setInput(path.activated[k], 0);
            pool->setInput(path.pooled.back(), 1);
            pool->setInput(path.taken.back(), 2);
            sequence.push_back(pool);
            x = path.pooled.back();
        } else {
            path.pooled.push_back(nullptr);
            path.taken.push_back(nullptr);
        }
    }
}

void VggLoss::upload() {
    for (size_t k = 0; k < weights.size(); ++k) {
        const auto base = std::filesystem::path(directory) / ("conv" + std::to_string(k));
        weights[k]->getDataBuffer(0).memcopyFrom(readFloats(base.string() + ".weight.bin"));
        biases[k]->getDataBuffer(0).memcopyFrom(readFloats(base.string() + ".bias.bin"));
    }
}

float VggLoss::loss() const {
    float sum = 0.0f;
    for (const auto& total : totals) {
        std::vector<float> value(1);
        total->getDataBuffer(0).memcopyTo(value);
        sum += value[0];
    }
    return sum;
}

} // namespace klartraum::neural
