// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/neural/photo_enhancer_training.hpp"

#include <stdexcept>

#include "klartraum/layers/layers.hpp"

namespace klartraum::neural {

namespace {

constexpr uint32_t kGBufferChannels = 11;

enum Group { Network = 0, Texture = 1, Codes = 2 };

layers::ConvAttributes convAttributes(uint32_t kernel, uint32_t stride) {
    layers::ConvAttributes attributes;
    attributes.kernelShape = {kernel, kernel};
    attributes.strides = {stride, stride};
    const uint32_t pad = kernel / 2;
    attributes.pads = {pad, pad, pad, pad};
    return attributes;
}

} // namespace

struct PhotoEnhancerTraining::Parameter {
    std::string name;
    int group = Network;
    std::shared_ptr<TensorElement<float>> value, gradient, firstMoment, secondMoment, average;
};

// One U-Net block: conv 3 x 3 (stride 1 or 2), ChannelNorm, ELU, conv 3 x 3, ChannelNorm, ELU, then FiLM by the
// appearance code; its activations and their gradients.
struct PhotoEnhancerTraining::Block {
    std::string prefix; ///< e.g. "generator.down.0."
    uint32_t inputChannels = 0, outputChannels = 0, inputSize = 0, outputSize = 0, stride = 1;
    uint32_t filmIndex = 0; ///< the block's FiLM heads
    std::shared_ptr<TensorElement<float>> in, a1, n1, stats1, e1, a2, n2, stats2, e2, out;
    std::shared_ptr<TensorElement<float>> scale, shift; ///< FiLM, [batch][outputChannels]
    std::shared_ptr<TensorElement<float>> dOut, de2, dn2, da2, de1, dn1, da1, dIn, dScale, dShift;
};

PhotoEnhancerTraining::PhotoEnhancerTraining(VulkanContext& vulkanContext, const PhotoEnhancerTrainingConfig& config)
    : vulkanContext(vulkanContext),
      config(config) {
    const uint32_t b = config.batch, s = config.size, levels = uint32_t(config.widths.size());
    if (levels < 2 || s % (1u << (levels - 1)) != 0 || s < 8u)
        throw std::invalid_argument("PhotoEnhancerTraining: the crop size must halve exactly at every level");
    this->config.texture.width = this->config.texture.height = s;
    this->config.texture.batch = b;
    this->config.texture.outputOffset = kGBufferChannels;
    const HashTextureShape& shape = this->config.texture;
    const uint32_t features = shape.levels * shape.features, channelsIn = kGBufferChannels + features;
    const bool film = config.appearances > 0;
    const auto& w = config.widths;

    // Parameters, in the order of the PyTorch state dict.
    addParameter("texture.table", {uint32_t(shape.classes.size()), shape.levels, shape.tableSize, shape.features},
                 Texture);
    const auto addBlock = [&](const std::string& prefix, uint32_t cin, uint32_t cout, uint32_t sizeIn,
                              uint32_t stride) {
        addParameter(prefix + "0.weight", {cout, cin, 3, 3}, Network);
        addParameter(prefix + "0.bias", {cout}, Network);
        addParameter(prefix + "1.weight", {cout}, Network);
        addParameter(prefix + "1.bias", {cout}, Network);
        addParameter(prefix + "3.weight", {cout, cout, 3, 3}, Network);
        addParameter(prefix + "3.bias", {cout}, Network);
        addParameter(prefix + "4.weight", {cout}, Network);
        addParameter(prefix + "4.bias", {cout}, Network);
        Block block;
        block.prefix = prefix;
        block.inputChannels = cin;
        block.outputChannels = cout;
        block.inputSize = sizeIn;
        block.outputSize = sizeIn / stride;
        block.stride = stride;
        block.filmIndex = uint32_t(blocks.size());
        blocks.push_back(block);
    };
    for (uint32_t i = 0; i < levels; ++i)
        addBlock("generator.down." + std::to_string(i) + ".", i == 0 ? channelsIn : w[i - 1], w[i],
                 i == 0 ? s : s >> (i - 1), i == 0 ? 1 : 2);
    for (uint32_t u = 0; u + 1 < levels; ++u) {
        const uint32_t deep = levels - 1 - u;
        addBlock("generator.up." + std::to_string(u) + ".", w[deep] + w[deep - 1], w[deep - 1], s >> (deep - 1), 1);
    }
    if (film) {
        addParameter("generator.film.mlp.0.weight", {config.filmHidden, config.appearanceSize}, Network);
        addParameter("generator.film.mlp.0.bias", {config.filmHidden}, Network);
        addParameter("generator.film.mlp.2.weight", {config.filmHidden, config.filmHidden}, Network);
        addParameter("generator.film.mlp.2.bias", {config.filmHidden}, Network);
        for (const char* heads : {"gammas", "betas"})
            for (size_t k = 0; k < blocks.size(); ++k) {
                const std::string name = "generator.film." + std::string(heads) + "." + std::to_string(k);
                addParameter(name + ".weight", {blocks[k].outputChannels, config.filmHidden}, Network);
                addParameter(name + ".bias", {blocks[k].outputChannels}, Network);
            }
    }
    // PyTorch registers the output convolution after FiLM, the codes after the generator.
    addParameter("generator.out.weight", {3, w[0], 1, 1}, Network);
    addParameter("generator.out.bias", {3}, Network);
    if (film)
        addParameter("codes.weight", {config.appearances, config.appearanceSize}, Codes);

    // The batch and the generator's input.
    channels = tensor({b, kGBufferChannels, s, s});
    coordinates = tensor({b, 3, s, s});
    classes = tensor({b, s, s});
    targets = tensor({b, 3, s, s});
    rows = tensor({b});
    input = tensor({b, channelsIn, s, s});
    scalesDummy = tensor({1});
    stepCounter = vulkanContext.create<TensorElement<uint32_t>>(std::vector<uint32_t>{1});

    // Forward: G-buffer channels and texture features into the input.
    {
        auto copy =
            training::channelCopy(vulkanContext, b, s * s, {kGBufferChannels, 0}, {channelsIn, 0}, kGBufferChannels);
        copy->setInput(channels, 0);
        copy->setInput(input, 1);
        add(copy);
        auto texture = hashTexture(vulkanContext, shape);
        texture->setInput(coordinates, 0);
        texture->setInput(classes, 1);
        texture->setInput(find("texture.table").value, 2);
        texture->setInput(input, 3);
        add(texture);
    }
    // FiLM: the codes through the MLP.
    const uint32_t d = config.appearanceSize, hiddenSize = config.filmHidden;
    if (film) {
        code = tensor({b, d});
        hidden1Pre = tensor({b, hiddenSize});
        hidden1 = tensor({b, hiddenSize});
        hidden2Pre = tensor({b, hiddenSize});
        hidden2 = tensor({b, hiddenSize});
        codeGradient = tensor({b, d});
        hidden1Gradient = tensor({b, hiddenSize});
        hidden1PreGradient = tensor({b, hiddenSize});
        hidden2Gradient = tensor({b, hiddenSize});
        hidden2PreGradient = tensor({b, hiddenSize});
        auto pick = training::embedding(vulkanContext, b, config.appearances, d);
        pick->setInput(find("codes.weight").value, 0);
        pick->setInput(rows, 1);
        pick->setInput(code, 2);
        add(pick);
        const auto linearLayer = [&](const std::string& name, std::shared_ptr<TensorElement<float>> x, uint32_t in,
                                     uint32_t out, std::shared_ptr<TensorElement<float>> y, float offset) {
            auto layer = training::linear(vulkanContext, {b, in, out}, offset);
            layer->setInput(x, 0);
            layer->setInput(find(name + ".weight").value, 1);
            layer->setInput(find(name + ".bias").value, 2);
            layer->setInput(y, 3);
            add(layer);
        };
        const auto eluLayer = [&](std::shared_ptr<TensorElement<float>> x, std::shared_ptr<TensorElement<float>> y) {
            auto layer = layers::elu(vulkanContext, {b, hiddenSize}, 1.0f);
            layer->setInput(x, 0);
            layer->setInput(y, 1);
            add(layer);
        };
        linearLayer("generator.film.mlp.0", code, d, hiddenSize, hidden1Pre, 0.0f);
        eluLayer(hidden1Pre, hidden1);
        linearLayer("generator.film.mlp.2", hidden1, hiddenSize, hiddenSize, hidden2Pre, 0.0f);
        eluLayer(hidden2Pre, hidden2);
        for (auto& block : blocks) {
            const std::string k = std::to_string(block.filmIndex);
            block.scale = tensor({b, block.outputChannels});
            block.shift = tensor({b, block.outputChannels});
            linearLayer("generator.film.gammas." + k, hidden2, hiddenSize, block.outputChannels, block.scale, 1.0f);
            linearLayer("generator.film.betas." + k, hidden2, hiddenSize, block.outputChannels, block.shift, 0.0f);
        }
    }
    // The U-Net.
    for (uint32_t i = 0; i < levels; ++i) {
        Block& block = blocks[i];
        block.in = i == 0 ? input : blocks[i - 1].out;
        forwardBlock(block);
    }
    std::shared_ptr<TensorElement<float>> x = blocks[levels - 1].out;
    for (uint32_t u = 0; u + 1 < levels; ++u) {
        Block& block = blocks[levels + u];
        const uint32_t deep = levels - 1 - u, size = s >> (deep - 1);
        auto upsampled = tensor({b, w[deep], size, size});
        auto upsample = layers::resizeLinear(vulkanContext, {b, w[deep], size / 2, size / 2}, {b, w[deep], size, size});
        upsample->setInput(x, 0);
        upsample->setInput(scalesDummy, 1);
        upsample->setInput(upsampled, 2);
        add(upsample);
        block.in = tensor({b, block.inputChannels, size, size});
        auto first =
            training::channelCopy(vulkanContext, b, size * size, {w[deep], 0}, {block.inputChannels, 0}, w[deep]);
        first->setInput(upsampled, 0);
        first->setInput(block.in, 1);
        add(first);
        auto second = training::channelCopy(vulkanContext, b, size * size, {w[deep - 1], 0},
                                            {block.inputChannels, w[deep]}, w[deep - 1]);
        second->setInput(blocks[deep - 1].out, 0);
        second->setInput(block.in, 1);
        add(second);
        forwardBlock(block);
        x = block.out;
    }
    logits = tensor({b, 3, s, s});
    output = tensor({b, 3, s, s});
    {
        auto conv =
            layers::conv(vulkanContext, convAttributes(1, 1), {b, w[0], s, s}, {3, w[0], 1, 1}, {3}, {b, 3, s, s});
        conv->setInput(x, 0);
        conv->setInput(find("generator.out.weight").value, 1);
        conv->setInput(find("generator.out.bias").value, 2);
        conv->setInput(logits, 3);
        add(conv);
        auto sigmoid = layers::unary(vulkanContext, layers::UnaryOp::Sigmoid, {b, 3, s, s});
        sigmoid->setInput(logits, 0);
        sigmoid->setInput(output, 1);
        add(sigmoid);
    }

    // The loss.
    outputGradient = tensor({b, 3, s, s});
    logitGradient = tensor({b, 3, s, s});
    lossShares = tensor({b, 3, s, s});
    lossTotal = tensor({1});
    {
        auto loss = training::photometricLoss(vulkanContext, {b, 3, s, s}, config.l1, config.colour);
        loss->setInput(output, 0);
        loss->setInput(targets, 1);
        loss->setInput(outputGradient, 2);
        loss->setInput(lossShares, 3);
        add(loss);
        auto total = training::sum(vulkanContext, b * 3 * s * s);
        total->setInput(lossShares, 0);
        total->setInput(lossTotal, 1);
        add(total);
    }

    // Backward: output layer, up blocks, down blocks, texture, FiLM.
    std::shared_ptr<TensorElement<float>> dx = tensor({b, w[0], s, s});
    {
        auto sigmoid = training::activationBackward(vulkanContext, training::Activation::Sigmoid, {b, 3, s, s});
        sigmoid->setInput(logits, 0);
        sigmoid->setInput(outputGradient, 1);
        sigmoid->setInput(logitGradient, 2);
        add(sigmoid);
        const auto attributes = convAttributes(1, 1);
        auto weights =
            training::convBackwardWeights(vulkanContext, attributes, {b, w[0], s, s}, {3, w[0], 1, 1}, {b, 3, s, s});
        weights->setInput(x, 0);
        weights->setInput(logitGradient, 1);
        weights->setInput(find("generator.out.weight").gradient, 2);
        weights->setInput(find("generator.out.bias").gradient, 3);
        add(weights);
        auto inputs =
            training::convBackwardInput(vulkanContext, attributes, {b, w[0], s, s}, {3, w[0], 1, 1}, {b, 3, s, s});
        inputs->setInput(logitGradient, 0);
        inputs->setInput(find("generator.out.weight").value, 1);
        inputs->setInput(dx, 2);
        add(inputs);
    }
    for (uint32_t i = 0; i < levels; ++i)
        blocks[i].dOut = tensor({b, blocks[i].outputChannels, blocks[i].outputSize, blocks[i].outputSize});
    for (int u = int(levels) - 2; u >= 0; --u) {
        Block& block = blocks[levels + uint32_t(u)];
        block.dOut = dx;
        backwardBlock(block);
        const uint32_t deep = levels - 1 - uint32_t(u), size = s >> (deep - 1);
        // The upsampled part's gradient back to the deeper output, the skip's to its down block.
        auto upsampledGradient = tensor({b, w[deep], size, size});
        auto first =
            training::channelCopy(vulkanContext, b, size * size, {block.inputChannels, 0}, {w[deep], 0}, w[deep]);
        first->setInput(block.dIn, 0);
        first->setInput(upsampledGradient, 1);
        add(first);
        dx = u == 0 ? blocks[levels - 1].dOut : tensor({b, w[deep], size / 2, size / 2});
        auto upsample = training::upsample2xBackward(vulkanContext, {b, w[deep], size / 2, size / 2});
        upsample->setInput(upsampledGradient, 0);
        upsample->setInput(dx, 1);
        add(upsample);
        auto skip = training::channelCopy(vulkanContext, b, size * size, {block.inputChannels, w[deep]},
                                          {w[deep - 1], 0}, w[deep - 1]);
        skip->setInput(block.dIn, 0);
        skip->setInput(blocks[deep - 1].dOut, 1);
        add(skip);
    }
    for (int i = int(levels) - 1; i >= 0; --i) {
        Block& block = blocks[uint32_t(i)];
        backwardBlock(block);
        if (i > 0) {
            // The down block's input is the previous one's output, which also fed a skip.
            const Block& previous = blocks[uint32_t(i) - 1];
            auto accumulate = training::channelCopy(vulkanContext, b, previous.outputSize * previous.outputSize,
                                                    {previous.outputChannels, 0}, {previous.outputChannels, 0},
                                                    previous.outputChannels, true);
            accumulate->setInput(block.dIn, 0);
            accumulate->setInput(previous.dOut, 1);
            add(accumulate);
        }
    }
    {
        const auto& table = find("texture.table");
        auto zero = training::fill(vulkanContext, uint32_t(table.gradient->getDataElementCount()));
        zero->setInput(table.gradient, 0);
        add(zero);
        auto texture = hashTextureBackward(vulkanContext, shape);
        texture->setInput(coordinates, 0);
        texture->setInput(classes, 1);
        texture->setInput(blocks[0].dIn, 2);
        texture->setInput(table.gradient, 3);
        add(texture);
    }
    if (film) {
        bool first = true;
        for (const auto& block : blocks)
            for (const auto& [heads, gradient] :
                 {std::pair{"gammas", block.dScale}, std::pair{"betas", block.dShift}}) {
                const std::string name = "generator.film." + std::string(heads) + "." + std::to_string(block.filmIndex);
                const training::LinearShape linearShape{b, hiddenSize, block.outputChannels};
                auto weights = training::linearBackwardWeights(vulkanContext, linearShape);
                weights->setInput(hidden2, 0);
                weights->setInput(gradient, 1);
                weights->setInput(find(name + ".weight").gradient, 2);
                weights->setInput(find(name + ".bias").gradient, 3);
                add(weights);
                auto inputs = training::linearBackwardInput(vulkanContext, linearShape, !first);
                inputs->setInput(gradient, 0);
                inputs->setInput(find(name + ".weight").value, 1);
                inputs->setInput(hidden2Gradient, 2);
                add(inputs);
                first = false;
            }
        const auto eluBackward = [&](std::shared_ptr<TensorElement<float>> pre,
                                     std::shared_ptr<TensorElement<float>> gradient,
                                     std::shared_ptr<TensorElement<float>> result) {
            auto layer = training::activationBackward(vulkanContext, training::Activation::Elu, {b, hiddenSize});
            layer->setInput(pre, 0);
            layer->setInput(gradient, 1);
            layer->setInput(result, 2);
            add(layer);
        };
        const auto linearBackward = [&](const std::string& name, std::shared_ptr<TensorElement<float>> x,
                                        std::shared_ptr<TensorElement<float>> gradient, uint32_t in,
                                        std::shared_ptr<TensorElement<float>> inputGradientOut) {
            const training::LinearShape linearShape{b, in, hiddenSize};
            auto weights = training::linearBackwardWeights(vulkanContext, linearShape);
            weights->setInput(x, 0);
            weights->setInput(gradient, 1);
            weights->setInput(find(name + ".weight").gradient, 2);
            weights->setInput(find(name + ".bias").gradient, 3);
            add(weights);
            auto inputs = training::linearBackwardInput(vulkanContext, linearShape);
            inputs->setInput(gradient, 0);
            inputs->setInput(find(name + ".weight").value, 1);
            inputs->setInput(inputGradientOut, 2);
            add(inputs);
        };
        eluBackward(hidden2Pre, hidden2Gradient, hidden2PreGradient);
        linearBackward("generator.film.mlp.2", hidden1, hidden2PreGradient, hiddenSize, hidden1Gradient);
        eluBackward(hidden1Pre, hidden1Gradient, hidden1PreGradient);
        linearBackward("generator.film.mlp.0", code, hidden1PreGradient, d, codeGradient);
        const auto& codes = find("codes.weight");
        auto scatter = training::embeddingBackward(vulkanContext, b, config.appearances, d, config.appearanceDecay);
        scatter->setInput(codes.value, 0);
        scatter->setInput(rows, 1);
        scatter->setInput(codeGradient, 2);
        scatter->setInput(codes.gradient, 3);
        add(scatter);
    }

    // The generator's gradients clipped to a global norm, then Adam and the average for every parameter.
    normTotal = tensor({1});
    {
        auto zero = training::fill(vulkanContext, 1);
        zero->setInput(normTotal, 0);
        add(zero);
        for (const auto& parameter : parameters)
            if (parameter.group == Network) {
                auto norm = training::squaredNorm(vulkanContext, uint32_t(parameter.gradient->getDataElementCount()));
                norm->setInput(parameter.gradient, 0);
                norm->setInput(normTotal, 1);
                add(norm);
            }
        for (const auto& parameter : parameters)
            if (parameter.group == Network) {
                auto clip = training::clipGradient(vulkanContext, uint32_t(parameter.gradient->getDataElementCount()),
                                                   config.clip);
                clip->setInput(parameter.gradient, 0);
                clip->setInput(normTotal, 1);
                add(clip);
            }
        auto counter = training::stepCounter(vulkanContext);
        counter->setInput(stepCounter, 0);
        add(counter);
        for (const auto& parameter : parameters) {
            const auto& adam = parameter.group == Network   ? config.networkAdam
                               : parameter.group == Texture ? config.textureAdam
                                                            : config.codesAdam;
            const uint32_t count = uint32_t(parameter.value->getDataElementCount());
            auto update = training::adamStep(vulkanContext, count, adam);
            update->setInput(parameter.value, 0);
            update->setInput(parameter.gradient, 1);
            update->setInput(parameter.firstMoment, 2);
            update->setInput(parameter.secondMoment, 3);
            update->setInput(stepCounter, 4);
            add(update);
            auto average = training::ema(vulkanContext, count, config.emaRate, config.emaWarmup);
            average->setInput(parameter.average, 0);
            average->setInput(parameter.value, 1);
            average->setInput(stepCounter, 2);
            add(average);
        }
    }
    graph = std::make_unique<ComputeGraph>(vulkanContext, 1);
    graph->compileFrom(sequence.back());
    // The buffers exist once the graph is compiled: parameters, moments and averages start at zero.
    for (const auto& parameter : parameters) {
        const std::vector<float> zeros(parameter.value->getDataElementCount(), 0.0f);
        for (const auto& t :
             {parameter.value, parameter.gradient, parameter.firstMoment, parameter.secondMoment, parameter.average})
            t->getDataBuffer(0).memcopyFrom(zeros);
    }
    stepCounter->getDataBuffer(0).memcopyFrom(std::vector<uint32_t>{0u});
}

PhotoEnhancerTraining::~PhotoEnhancerTraining() = default;

std::shared_ptr<TensorElement<float>> PhotoEnhancerTraining::tensor(const std::vector<uint32_t>& shape) {
    return vulkanContext.create<TensorElement<float>>(shape);
}

PhotoEnhancerTraining::Parameter& PhotoEnhancerTraining::addParameter(const std::string& name,
                                                                      const std::vector<uint32_t>& shape, int group) {
    Parameter parameter;
    parameter.name = name;
    parameter.group = group;
    parameter.value = tensor(shape);
    parameter.gradient = tensor(shape);
    parameter.firstMoment = tensor(shape);
    parameter.secondMoment = tensor(shape);
    parameter.average = tensor(shape);
    parameterIndex[name] = parameters.size();
    parameters.push_back(parameter);
    return parameters.back();
}

const PhotoEnhancerTraining::Parameter& PhotoEnhancerTraining::find(const std::string& name) const {
    const auto it = parameterIndex.find(name);
    if (it == parameterIndex.end())
        throw std::invalid_argument("PhotoEnhancerTraining: no parameter " + name);
    return parameters[it->second];
}

void PhotoEnhancerTraining::add(ComputeGraphElementPtr element) {
    if (!sequence.empty())
        element->addDependency(sequence.back());
    sequence.push_back(element);
}

void PhotoEnhancerTraining::forwardBlock(Block& block) {
    const uint32_t b = config.batch, cin = block.inputChannels, cout = block.outputChannels;
    const uint32_t si = block.inputSize, so = block.outputSize;
    const std::vector<uint32_t> shape{b, cout, so, so};
    block.a1 = tensor(shape);
    block.n1 = tensor(shape);
    block.stats1 = tensor({b, 2, so, so});
    block.e1 = tensor(shape);
    block.a2 = tensor(shape);
    block.n2 = tensor(shape);
    block.stats2 = tensor({b, 2, so, so});
    block.e2 = tensor(shape);
    block.out = block.scale ? tensor(shape) : block.e2;
    const auto conv = [&](const std::string& name, std::shared_ptr<TensorElement<float>> x, uint32_t channelsIn,
                          uint32_t sizeIn, uint32_t stride, std::shared_ptr<TensorElement<float>> y) {
        auto layer = layers::conv(vulkanContext, convAttributes(3, stride), {b, channelsIn, sizeIn, sizeIn},
                                  {cout, channelsIn, 3, 3}, {cout}, shape);
        layer->setInput(x, 0);
        layer->setInput(find(name + ".weight").value, 1);
        layer->setInput(find(name + ".bias").value, 2);
        layer->setInput(y, 3);
        add(layer);
    };
    const auto norm = [&](const std::string& name, std::shared_ptr<TensorElement<float>> x,
                          std::shared_ptr<TensorElement<float>> y, std::shared_ptr<TensorElement<float>> stats) {
        auto layer = training::channelNorm(vulkanContext, shape);
        layer->setInput(x, 0);
        layer->setInput(find(name + ".weight").value, 1);
        layer->setInput(find(name + ".bias").value, 2);
        layer->setInput(y, 3);
        layer->setInput(stats, 4);
        add(layer);
    };
    const auto elu = [&](std::shared_ptr<TensorElement<float>> x, std::shared_ptr<TensorElement<float>> y) {
        auto layer = layers::elu(vulkanContext, shape, 1.0f);
        layer->setInput(x, 0);
        layer->setInput(y, 1);
        add(layer);
    };
    conv(block.prefix + "0", block.in, cin, si, block.stride, block.a1);
    norm(block.prefix + "1", block.a1, block.n1, block.stats1);
    elu(block.n1, block.e1);
    conv(block.prefix + "3", block.e1, cout, so, 1, block.a2);
    norm(block.prefix + "4", block.a2, block.n2, block.stats2);
    elu(block.n2, block.e2);
    if (block.scale) {
        auto modulate = training::film(vulkanContext, shape);
        modulate->setInput(block.e2, 0);
        modulate->setInput(block.scale, 1);
        modulate->setInput(block.shift, 2);
        modulate->setInput(block.out, 3);
        add(modulate);
    }
}

void PhotoEnhancerTraining::backwardBlock(Block& block) {
    const uint32_t b = config.batch, cin = block.inputChannels, cout = block.outputChannels;
    const uint32_t si = block.inputSize, so = block.outputSize;
    const std::vector<uint32_t> shape{b, cout, so, so};
    block.dn2 = tensor(shape);
    block.da2 = tensor(shape);
    block.de1 = tensor(shape);
    block.dn1 = tensor(shape);
    block.da1 = tensor(shape);
    block.dIn = tensor({b, cin, si, si});
    if (block.scale) {
        block.de2 = tensor(shape);
        block.dScale = tensor({b, cout});
        block.dShift = tensor({b, cout});
        auto modulate = training::filmBackward(vulkanContext, shape);
        modulate->setInput(block.e2, 0);
        modulate->setInput(block.scale, 1);
        modulate->setInput(block.dOut, 2);
        modulate->setInput(block.de2, 3);
        modulate->setInput(block.dScale, 4);
        modulate->setInput(block.dShift, 5);
        add(modulate);
    } else {
        block.de2 = block.dOut;
    }
    const auto elu = [&](std::shared_ptr<TensorElement<float>> pre, std::shared_ptr<TensorElement<float>> gradient,
                         std::shared_ptr<TensorElement<float>> result) {
        auto layer = training::activationBackward(vulkanContext, training::Activation::Elu, shape);
        layer->setInput(pre, 0);
        layer->setInput(gradient, 1);
        layer->setInput(result, 2);
        add(layer);
    };
    const auto norm = [&](const std::string& name, std::shared_ptr<TensorElement<float>> x,
                          std::shared_ptr<TensorElement<float>> stats, std::shared_ptr<TensorElement<float>> gradient,
                          std::shared_ptr<TensorElement<float>> result) {
        auto inputs = training::channelNormBackwardInput(vulkanContext, shape);
        inputs->setInput(x, 0);
        inputs->setInput(find(name + ".weight").value, 1);
        inputs->setInput(stats, 2);
        inputs->setInput(gradient, 3);
        inputs->setInput(result, 4);
        add(inputs);
        auto params = training::channelNormBackwardParameters(vulkanContext, shape);
        params->setInput(x, 0);
        params->setInput(stats, 1);
        params->setInput(gradient, 2);
        params->setInput(find(name + ".weight").gradient, 3);
        params->setInput(find(name + ".bias").gradient, 4);
        add(params);
    };
    const auto conv = [&](const std::string& name, std::shared_ptr<TensorElement<float>> x, uint32_t channelsIn,
                          uint32_t sizeIn, uint32_t stride, std::shared_ptr<TensorElement<float>> gradient,
                          std::shared_ptr<TensorElement<float>> result) {
        const auto attributes = convAttributes(3, stride);
        const std::vector<uint32_t> inShape{b, channelsIn, sizeIn, sizeIn}, weightShape{cout, channelsIn, 3, 3};
        auto weights = training::convBackwardWeights(vulkanContext, attributes, inShape, weightShape, shape);
        weights->setInput(x, 0);
        weights->setInput(gradient, 1);
        weights->setInput(find(name + ".weight").gradient, 2);
        weights->setInput(find(name + ".bias").gradient, 3);
        add(weights);
        auto inputs = training::convBackwardInput(vulkanContext, attributes, inShape, weightShape, shape);
        inputs->setInput(gradient, 0);
        inputs->setInput(find(name + ".weight").value, 1);
        inputs->setInput(result, 2);
        add(inputs);
    };
    elu(block.n2, block.de2, block.dn2);
    norm(block.prefix + "4", block.a2, block.stats2, block.dn2, block.da2);
    conv(block.prefix + "3", block.e1, cout, so, 1, block.da2, block.de1);
    elu(block.n1, block.de1, block.dn1);
    norm(block.prefix + "1", block.a1, block.stats1, block.dn1, block.da1);
    conv(block.prefix + "0", block.in, cin, si, block.stride, block.da1, block.dIn);
}

void PhotoEnhancerTraining::setBatch(const std::vector<float>& channelValues,
                                     const std::vector<float>& coordinateValues, const std::vector<float>& classValues,
                                     const std::vector<float>& targetValues, const std::vector<float>& rowValues) {
    channels->getDataBuffer(0).memcopyFrom(channelValues);
    coordinates->getDataBuffer(0).memcopyFrom(coordinateValues);
    classes->getDataBuffer(0).memcopyFrom(classValues);
    targets->getDataBuffer(0).memcopyFrom(targetValues);
    if (config.appearances > 0)
        rows->getDataBuffer(0).memcopyFrom(rowValues);
}

void PhotoEnhancerTraining::step() { graph->submitAndWait(vulkanContext.getGraphicsQueue(), 0); }

float PhotoEnhancerTraining::loss() const {
    std::vector<float> value(1);
    lossTotal->getDataBuffer(0).memcopyTo(value);
    return value[0];
}

std::vector<float> PhotoEnhancerTraining::image() const {
    std::vector<float> values(output->getDataElementCount());
    output->getDataBuffer(0).memcopyTo(values);
    return values;
}

std::vector<std::string> PhotoEnhancerTraining::parameterNames() const {
    std::vector<std::string> names;
    for (const auto& parameter : parameters)
        names.push_back(parameter.name);
    return names;
}

std::shared_ptr<TensorElement<float>> PhotoEnhancerTraining::parameter(const std::string& name) const {
    return find(name).value;
}

std::shared_ptr<TensorElement<float>> PhotoEnhancerTraining::gradient(const std::string& name) const {
    return find(name).gradient;
}

std::shared_ptr<TensorElement<float>> PhotoEnhancerTraining::average(const std::string& name) const {
    return find(name).average;
}

void PhotoEnhancerTraining::setParameter(const std::string& name, const std::vector<float>& values) {
    const auto& parameter = find(name);
    if (values.size() != parameter.value->getDataElementCount())
        throw std::invalid_argument("PhotoEnhancerTraining: " + name + " has another size");
    parameter.value->getDataBuffer(0).memcopyFrom(values);
    parameter.average->getDataBuffer(0).memcopyFrom(values);
}

} // namespace klartraum::neural
