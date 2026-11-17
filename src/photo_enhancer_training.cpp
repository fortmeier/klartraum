// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/neural/photo_enhancer_training.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <tuple>

#include "klartraum/layers/layers.hpp"

namespace klartraum::neural {

namespace {

enum Group { Network = 0, Texture = 1, Codes = 2, Discriminator = 3 };

// The PatchGAN's layers: widths, strides (all 4 x 4, padding 1), then the logits.
constexpr uint32_t kCriticWidths[4] = {64, 128, 256, 512};
constexpr uint32_t kCriticStrides[4] = {2, 2, 2, 1};

// The G-buffer channels the discriminator sees in front of the image's three.
uint32_t criticConditioning(const PhotoEnhancerTrainingConfig& config) {
    if (!config.conditionalDiscriminator)
        return 0u;
    return config.criticGBufferChannels > 0 ? std::min(config.criticGBufferChannels, config.gbufferChannels)
                                            : config.gbufferChannels;
}

layers::ConvAttributes convAttributes(uint32_t kernel, uint32_t stride) {
    layers::ConvAttributes attributes;
    attributes.kernelShape = {kernel, kernel};
    attributes.strides = {stride, stride};
    const uint32_t pad = kernel / 2;
    attributes.pads = {pad, pad, pad, pad};
    return attributes;
}

} // namespace

// One pass of the 2-scale PatchGAN over a batch of pairs: per scale its input (the pairs, or their 2 x 2 average), the
// input times 2 minus 1, every layer's convolution and leaky ReLU, the logits; and their gradients.
struct PhotoEnhancerTraining::Critic {
    uint32_t batch = 0;
    struct Scale {
        uint32_t size = 0;
        std::shared_ptr<TensorElement<float>> input, scaled, logits, logitGradient, scaledGradient, inputGradient;
        std::vector<std::shared_ptr<TensorElement<float>>> pre, activated, preGradient, activatedGradient;
        std::vector<uint32_t> sizes; ///< of each layer's output, then the logits'
    };
    std::vector<Scale> scales;
};

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
    if ((config.latentCheck > 0.0f || config.temporal > 0.0f) && (config.latentSize == 0 || config.appearances > 0))
        throw std::invalid_argument(
            "PhotoEnhancerTraining: the latent check and the temporal loss need latents and no appearance codes");
    // The PatchGAN's coarsest scale needs 24 x 24 pixels: three halvings and two 4 x 4 convolutions leave 1 x 1 logits.
    if (config.adversarial > 0.0f && !config.forwardOnly &&
        (config.criticScales == 0 || (s >> (config.criticScales - 1)) < 24u))
        throw std::invalid_argument("PhotoEnhancerTraining: the discriminator's coarsest scale is below 24 pixels");
    this->config.texture.width = this->config.texture.height = s;
    this->config.texture.batch = b;
    const uint32_t kGBufferChannels = config.gbufferChannels;
    this->config.texture.outputOffset = kGBufferChannels;
    const HashTextureShape& shape = this->config.texture;
    const bool mlp = config.textureMlp.frequencies > 0;
    if (mlp && (config.latentSize == 0 || shape.levels > 0 || shape.classes.empty() || shape.classes.size() > 4))
        throw std::invalid_argument(
            "PhotoEnhancerTraining: the MLP texture needs latents, a texture of no levels and 1 to 4 classes");
    if (config.latentCodes > 0 && (config.latentSize == 0 || config.appearances > 0 || config.randomLatents ||
                                   config.encodeLatents || config.latentCheck > 0.0f))
        throw std::invalid_argument("PhotoEnhancerTraining: the latent table needs latents, no appearance codes, no "
                                    "given or encoded latents and no latent check");
    if (!config.latentToDecoder && (config.latentSize == 0 || config.latentCheck > 0.0f))
        throw std::invalid_argument(
            "PhotoEnhancerTraining: latents only in the texture need latents and no latent check");
    const uint32_t features = textureFeatures();
    const uint32_t channelsIn = kGBufferChannels + features + config.latentSize;
    this->config.texture.outputChannels = channelsIn;
    const bool table = config.latentCodes > 0 && !config.forwardOnly;
    const bool basis = config.textureBasis.enabled();
    if (basis && (config.latentSize == 0 || shape.levels > 0 || mlp || config.latentCheck > 0.0f ||
                  shape.classes.empty() || shape.classes.size() > 4))
        throw std::invalid_argument("PhotoEnhancerTraining: the feature texture needs latents, a texture of no levels "
                                    "with 1 to 4 classes, no MLP texture and no latent check");
    const bool film = config.appearances > 0;
    const auto& w = config.widths;

    // Parameters, in the order of the PyTorch state dict.
    // A texture of no levels: the generator sees the G-buffer channels (and the latents) only.
    // With textureOff the texture's parameters are not created either (load() skips them).
    const bool textured = shape.levels > 0 && !config.textureOff;
    if (textured)
        addParameter("texture.table", {uint32_t(shape.classes.size()), shape.levels, shape.tableSize, shape.features},
                     Texture);
    if (basis && !config.textureOff)
        for (size_t level = 0; level < config.textureBasis.resolutions.size(); ++level)
            addParameter("texture.basis." + std::to_string(level), {uint32_t(basisShape(level, false).tableSize())},
                         Texture);
    if (mlp && !config.textureOff) {
        const auto& t = config.textureMlp;
        const auto addMlp = [&](const std::string& prefix, uint32_t frequencies, uint32_t width) {
            uint32_t previous = 3 * (2 * frequencies + 1) + config.latentSize;
            for (uint32_t layer = 0; layer <= t.hidden; ++layer) {
                const uint32_t next = layer == t.hidden ? t.outputs : width;
                const std::string name = prefix + std::to_string(2 * layer) + ".";
                addParameter(name + "weight", {next, previous, 1, 1}, Network);
                addParameter(name + "bias", {next}, Network);
                previous = next;
            }
        };
        addMlp("generator.texture.", t.frequencies, t.width);
        for (size_t j = 0; j < t.residual.size(); ++j)
            addMlp("generator.texture_residual." + std::to_string(j) + ".", t.residual[j].first, t.residual[j].second);
    }
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
    // The encoder, unless the latents are given.
    if (config.latentSize > 0 && (!config.forwardOnly || config.encodeLatents) && !config.randomLatents &&
        config.latentCodes == 0) {
        uint32_t previous = 3;
        for (size_t i = 0; i < config.encoderWidths.size(); ++i) {
            const std::string conv = "encoder.body." + std::to_string(3 * i) + ".";
            const std::string norm = "encoder.body." + std::to_string(3 * i + 1) + ".";
            addParameter(conv + "weight", {config.encoderWidths[i], previous, 3, 3}, Network);
            addParameter(conv + "bias", {config.encoderWidths[i]}, Network);
            addParameter(norm + "weight", {config.encoderWidths[i]}, Network);
            addParameter(norm + "bias", {config.encoderWidths[i]}, Network);
            previous = config.encoderWidths[i];
        }
        addParameter("encoder.head.weight", {2 * config.latentSize, previous, 1, 1}, Network);
        addParameter("encoder.head.bias", {2 * config.latentSize}, Network);
    }
    if (film)
        addParameter("codes.weight", {config.appearances, config.appearanceSize}, Codes);
    if (table)
        addParameter("latents.weight", {config.latentCodes, config.latentObjects * config.latentSize}, Codes);
    if (config.adversarial > 0.0f && !config.forwardOnly)
        for (uint32_t scale = 0; scale < config.criticScales; ++scale) {
            const std::string prefix = "discriminator.discriminators." + std::to_string(scale) + ".";
            uint32_t previous = criticConditioning(config) + 3;
            for (uint32_t layer = 0; layer < 4; ++layer) {
                const std::string name = prefix + "layers." + std::to_string(layer) + ".0.";
                addParameter(name + "weight", {kCriticWidths[layer], previous, 4, 4}, Discriminator);
                addParameter(name + "bias", {kCriticWidths[layer]}, Discriminator);
                previous = kCriticWidths[layer];
            }
            addParameter(prefix + "logits.weight", {1, previous, 4, 4}, Discriminator);
            addParameter(prefix + "logits.bias", {1}, Discriminator);
        }

    // The batch and the generator's input.
    channels = tensor({b, kGBufferChannels, s, s});
    coordinates = tensor({b, 3, s, s});
    classes = tensor({b, s, s});
    targets = tensor({b, 3, s, s});
    rows = tensor({b});
    input = tensor({b, channelsIn, s, s});
    scalesDummy = tensor({1});
    stepCounter = vulkanContext.create<TensorElement<uint32_t>>(std::vector<uint32_t>{1});

    // Forward: G-buffer channels and texture features into the input; without the texture its channels stay zero.
    if (config.textureOff) {
        if (!config.forwardOnly)
            throw std::invalid_argument("PhotoEnhancerTraining: textureOff is for rendering only");
        auto zero = training::fill(vulkanContext, b * channelsIn * s * s);
        zero->setInput(input, 0);
        add(zero);
    }
    {
        auto copy =
            training::channelCopy(vulkanContext, b, s * s, {kGBufferChannels, 0}, {channelsIn, 0}, kGBufferChannels);
        copy->setInput(channels, 0);
        copy->setInput(input, 1);
        add(copy);
        if (textured) {
            auto texture = hashTexture(vulkanContext, shape);
            texture->setInput(coordinates, 0);
            texture->setInput(classes, 1);
            texture->setInput(find("texture.table").value, 2);
            texture->setInput(input, 3);
            add(texture);
        }
    }
    // The latent G-buffer after the texture features: every object's latent at its pixels, from the encoder of the
    // photos (or, forward only, given).
    if (config.latentSize > 0) {
        const uint32_t k = config.latentObjects, d = config.latentSize;
        objects = tensor({b, s, s});
        counts = tensor({b, k});
        latentValues = tensor({b, k, d});
        if (table) {
            auto pick = training::embedding(vulkanContext, b, config.latentCodes, k * d);
            pick->setInput(find("latents.weight").value, 0);
            pick->setInput(rows, 1);
            pick->setInput(latentValues, 2);
            add(pick);
        } else if ((!config.forwardOnly || config.encodeLatents) && !config.randomLatents && config.latentCodes == 0)
            buildEncoder();
        auto splat = training::objectBroadcast(vulkanContext, {b, d, s, s}, k, {s, s, 1, 0}, channelsIn,
                                               kGBufferChannels + features);
        splat->setInput(latentValues, 0);
        splat->setInput(objects, 1);
        splat->setInput(counts, 2);
        splat->setInput(input, 3);
        add(splat);
        if (mlp && !config.textureOff)
            mainMlp = forwardTextureMlp(coordinates, classes, input);
        if (basis && !config.textureOff) {
            latentImage = tensor({b, d, s, s});
            auto image = training::objectBroadcast(vulkanContext, {b, d, s, s}, k, {s, s, 1, 0}, d, 0);
            image->setInput(latentValues, 0);
            image->setInput(objects, 1);
            image->setInput(counts, 2);
            image->setInput(latentImage, 3);
            add(image);
            forwardTextureBasis(coordinates, classes, latentImage, input);
        }
        if (!config.latentToDecoder)
            hideLatents(input);
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
    std::shared_ptr<TensorElement<float>> x;
    std::tie(x, logits, output) = forwardGenerator(blocks, input);

    if (config.forwardOnly) {
        graph = std::make_unique<ComputeGraph>(vulkanContext, 1);
        if (config.profile)
            graph->enableProfiling();
        graph->compileFrom(sequence.back());
        for (const auto& parameter : parameters) {
            const std::vector<float> zeros(parameter.value->getDataElementCount(), 0.0f);
            parameter.value->getDataBuffer(0).memcopyFrom(zeros);
        }
        return;
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
        // The perceptual loss adds its gradient to the photometric one.
        if (config.perceptual > 0.0f) {
            vgg = std::make_unique<VggLoss>(vulkanContext, config.vggDirectory, b, s, s, config.perceptual, output,
                                            targets, outputGradient);
            for (const auto& element : vgg->elements())
                add(element);
        }
        if (config.adversarial > 0.0f)
            buildDiscriminator();
    }

    // Backward: output layer, up blocks, down blocks, texture, FiLM.
    backwardGenerator(blocks, x, logits, outputGradient, logitGradient);
    if (textured) {
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
    if (mlp)
        mlpInputGradient = backwardTextureMlp(mainMlp, blocks[0].dIn);
    if (basis) {
        for (size_t level = 0; level < config.textureBasis.resolutions.size(); ++level) {
            const auto& gradient = find("texture.basis." + std::to_string(level)).gradient;
            auto zero = training::fill(vulkanContext, uint32_t(gradient->getDataElementCount()));
            zero->setInput(gradient, 0);
            add(zero);
        }
        basisLatentGradient = tensor({b, config.latentSize, s, s});
        backwardTextureBasis(coordinates, classes, latentImage, blocks[0].dIn, basisLatentGradient);
        const auto& t = config.textureBasis;
        if (t.decay > 0.0f || t.smoothness > 0.0f)
            for (size_t level = 0; level < t.resolutions.size(); ++level) {
                const float scale = float(t.resolutions[level]) / float(t.resolutions[0]);
                const auto& table = find("texture.basis." + std::to_string(level));
                auto regularize = training::textureBasisRegularize(vulkanContext, basisShape(level, false),
                                                                   scale * t.decay, scale * t.smoothness);
                regularize->setInput(table.value, 0);
                regularize->setInput(table.gradient, 1);
                add(regularize);
            }
    }
    if (config.latentSize > 0 && !config.randomLatents)
        buildLatentBackward();
    if (config.latentCheck > 0.0f)
        buildLatentCheck();
    if (config.temporal > 0.0f)
        buildTemporal();
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
            if (parameter.group == Discriminator)
                continue; // updated in the discriminator's step
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
    if (config.profile)
        graph->enableProfiling();
    graph->compileFrom(sequence.back());
    // The buffers exist once the graph is compiled: parameters, moments and averages start at zero (the
    // discriminator has no average).
    for (const auto& parameter : parameters) {
        const std::vector<float> zeros(parameter.value->getDataElementCount(), 0.0f);
        for (const auto& t : {parameter.value, parameter.gradient, parameter.firstMoment, parameter.secondMoment})
            t->getDataBuffer(0).memcopyFrom(zeros);
        if (parameter.group != Discriminator)
            parameter.average->getDataBuffer(0).memcopyFrom(zeros);
    }
    stepCounter->getDataBuffer(0).memcopyFrom(std::vector<uint32_t>{0u});
    if (vgg)
        vgg->upload();
    if (discriminatorCounter)
        discriminatorCounter->getDataBuffer(0).memcopyFrom(std::vector<uint32_t>{0u});
    random.seed(config.seed);
}

PhotoEnhancerTraining::~PhotoEnhancerTraining() = default;

void PhotoEnhancerTraining::buildEncoder() {
    const uint32_t b = config.batch, s = config.size, k = config.latentObjects, d = config.latentSize;
    const uint32_t size = s >> config.encoderWidths.size();
    // The means and log-variances per pixel, then per object (the object map sampled at the pixels' centres).
    encoderStatistics = tensor({b, 2 * d, size, size});
    encoderLayers = forwardEncoder(targets, encoderStatistics);
    const uint32_t step = s / size;
    pooled = tensor({b, k, 2 * d});
    auto pool = training::objectMean(vulkanContext, {b, 2 * d, size, size}, k, {s, s, step, step / 2});
    pool->setInput(encoderStatistics, 0);
    pool->setInput(objects, 1);
    pool->setInput(pooled, 2);
    pool->setInput(counts, 3);
    add(pool);
    latentNoise = tensor({b, k, d});
    klShares = tensor({b, k});
    klTotal = tensor({1});
    auto sample = training::vae(vulkanContext, b, k, d, config.klWeight, config.freeBits);
    sample->setInput(pooled, 0);
    sample->setInput(latentNoise, 1);
    sample->setInput(counts, 2);
    sample->setInput(latentValues, 3);
    sample->setInput(klShares, 4);
    add(sample);
    auto total = training::sum(vulkanContext, b * k);
    total->setInput(klShares, 0);
    total->setInput(klTotal, 1);
    add(total);
}

std::vector<PhotoEnhancerTraining::EncoderLayer>
PhotoEnhancerTraining::forwardEncoder(std::shared_ptr<TensorElement<float>> image,
                                      std::shared_ptr<TensorElement<float>> statistics) {
    const uint32_t b = config.batch, s = config.size, d = config.latentSize;
    std::vector<EncoderLayer> layers;
    auto x = image;
    uint32_t channels = 3, size = s;
    for (size_t i = 0; i < config.encoderWidths.size(); ++i) {
        EncoderLayer layer;
        layer.inputChannels = channels;
        layer.outputChannels = config.encoderWidths[i];
        layer.inputSize = size;
        layer.in = x;
        const uint32_t next = size / 2;
        const std::vector<uint32_t> shape{b, layer.outputChannels, next, next};
        layer.a = tensor(shape);
        layer.n = tensor(shape);
        layer.stats = tensor({b, 2, next, next});
        layer.e = tensor(shape);
        const std::string conv = "encoder.body." + std::to_string(3 * i) + ".";
        const std::string norm = "encoder.body." + std::to_string(3 * i + 1) + ".";
        auto convolution = layers::conv(vulkanContext, convAttributes(3, 2), {b, channels, size, size},
                                        {layer.outputChannels, channels, 3, 3}, {layer.outputChannels}, shape);
        convolution->setInput(x, 0);
        convolution->setInput(find(conv + "weight").value, 1);
        convolution->setInput(find(conv + "bias").value, 2);
        convolution->setInput(layer.a, 3);
        add(convolution);
        auto normalize = training::channelNorm(vulkanContext, shape);
        normalize->setInput(layer.a, 0);
        normalize->setInput(find(norm + "weight").value, 1);
        normalize->setInput(find(norm + "bias").value, 2);
        normalize->setInput(layer.n, 3);
        normalize->setInput(layer.stats, 4);
        add(normalize);
        auto elu = layers::elu(vulkanContext, shape, 1.0f);
        elu->setInput(layer.n, 0);
        elu->setInput(layer.e, 1);
        add(elu);
        layers.push_back(layer);
        x = layer.e;
        channels = layer.outputChannels;
        size = next;
    }
    auto head = layers::conv(vulkanContext, convAttributes(1, 1), {b, channels, size, size}, {2 * d, channels, 1, 1},
                             {2 * d}, {b, 2 * d, size, size});
    head->setInput(x, 0);
    head->setInput(find("encoder.head.weight").value, 1);
    head->setInput(find("encoder.head.bias").value, 2);
    head->setInput(statistics, 3);
    add(head);
    return layers;
}

std::shared_ptr<TensorElement<float>>
PhotoEnhancerTraining::backwardEncoderInput(std::vector<EncoderLayer>& layers,
                                            std::shared_ptr<TensorElement<float>> statisticsGradient) {
    const uint32_t b = config.batch, d = config.latentSize;
    const uint32_t size = config.size >> layers.size();
    const auto backwardConv = [&](const std::string& name, uint32_t channelsIn, uint32_t sizeIn, uint32_t channelsOut,
                                  uint32_t kernel, uint32_t stride, std::shared_ptr<TensorElement<float>> gradient,
                                  std::shared_ptr<TensorElement<float>> result) {
        const uint32_t sizeOut = sizeIn / stride;
        auto inputs =
            training::convBackwardInput(vulkanContext, convAttributes(kernel, stride), {b, channelsIn, sizeIn, sizeIn},
                                        {channelsOut, channelsIn, kernel, kernel}, {b, channelsOut, sizeOut, sizeOut});
        inputs->setInput(gradient, 0);
        inputs->setInput(find(name + "weight").value, 1);
        inputs->setInput(result, 2);
        add(inputs);
    };
    auto& last = layers.back();
    last.de = tensor({b, last.outputChannels, size, size});
    backwardConv("encoder.head.", last.outputChannels, size, 2 * d, 1, 1, statisticsGradient, last.de);
    std::shared_ptr<TensorElement<float>> imageGradient;
    for (int i = int(layers.size()) - 1; i >= 0; --i) {
        auto& layer = layers[size_t(i)];
        const uint32_t next = layer.inputSize / 2;
        const std::vector<uint32_t> shape{b, layer.outputChannels, next, next};
        layer.dn = tensor(shape);
        layer.da = tensor(shape);
        auto elu = training::activationBackward(vulkanContext, training::Activation::Elu, shape);
        elu->setInput(layer.n, 0);
        elu->setInput(layer.de, 1);
        elu->setInput(layer.dn, 2);
        add(elu);
        auto norm = training::channelNormBackwardInput(vulkanContext, shape);
        norm->setInput(layer.a, 0);
        norm->setInput(find("encoder.body." + std::to_string(3 * i + 1) + ".weight").value, 1);
        norm->setInput(layer.stats, 2);
        norm->setInput(layer.dn, 3);
        norm->setInput(layer.da, 4);
        add(norm);
        auto previous = tensor({b, layer.inputChannels, layer.inputSize, layer.inputSize});
        if (i > 0)
            layers[size_t(i) - 1].de = previous;
        else
            imageGradient = previous;
        backwardConv("encoder.body." + std::to_string(3 * i) + ".", layer.inputChannels, layer.inputSize,
                     layer.outputChannels, 3, 2, layer.da, previous);
    }
    return imageGradient;
}

void PhotoEnhancerTraining::buildLatentCheck() {
    const uint32_t b = config.batch, s = config.size, k = config.latentObjects, d = config.latentSize;
    const uint32_t gbuffer = config.gbufferChannels, features = textureFeatures();
    const uint32_t channelsIn = gbuffer + features + d, size = s >> config.encoderWidths.size(), step = s / size;
    // The targets: view i's latents are view permutation[i]'s sampled ones.
    latentPermutation = tensor({b});
    auto swapped = tensor({b, k, d});
    auto pick = training::embedding(vulkanContext, b, b, k * d);
    pick->setInput(latentValues, 0);
    pick->setInput(latentPermutation, 1);
    pick->setInput(swapped, 2);
    add(pick);
    // The second renders: the first input's G-buffer channels and texture features with the permuted latent G-buffer.
    auto input2 = tensor({b, channelsIn, s, s});
    auto copy = training::channelCopy(vulkanContext, b, s * s, {channelsIn, 0}, {channelsIn, 0}, gbuffer + features);
    copy->setInput(input, 0);
    copy->setInput(input2, 1);
    add(copy);
    auto splat =
        training::objectBroadcast(vulkanContext, {b, d, s, s}, k, {s, s, 1, 0}, channelsIn, gbuffer + features);
    splat->setInput(swapped, 0);
    splat->setInput(objects, 1);
    splat->setInput(counts, 2);
    splat->setInput(input2, 3);
    add(splat);
    const bool mlp = config.textureMlp.frequencies > 0;
    if (mlp)
        checkMlp = forwardTextureMlp(coordinates, classes, input2);
    checkBlocks = blocks;
    auto [x2, logits2, output2] = forwardGenerator(checkBlocks, input2);
    // The fixed encoder on them, its means per object against the targets.
    auto statistics2 = tensor({b, 2 * d, size, size});
    checkEncoderLayers = forwardEncoder(output2, statistics2);
    auto pooled2 = tensor({b, k, 2 * d}), counts2 = tensor({b, k});
    auto pool = training::objectMean(vulkanContext, {b, 2 * d, size, size}, k, {s, s, step, step / 2});
    pool->setInput(statistics2, 0);
    pool->setInput(objects, 1);
    pool->setInput(pooled2, 2);
    pool->setInput(counts2, 3);
    add(pool);
    auto means2 = tensor({b, k, d}), meansGradient = tensor({b, k, d}), shares = tensor({b * k * d});
    auto means = training::channelCopy(vulkanContext, b * k, 1, {2 * d, 0}, {d, 0}, d);
    means->setInput(pooled2, 0);
    means->setInput(means2, 1);
    add(means);
    auto distance = training::l1Distance(vulkanContext, b * k * d, config.latentCheck);
    distance->setInput(means2, 0);
    distance->setInput(swapped, 1);
    distance->setInput(meansGradient, 2);
    distance->setInput(shares, 3);
    add(distance);
    checkTotal = tensor({1});
    auto total = training::sum(vulkanContext, b * k * d);
    total->setInput(shares, 0);
    total->setInput(checkTotal, 1);
    add(total);
    // Back through the fixed encoder (the log-variances get no gradient) to the second renders.
    auto pooledGradient2 = tensor({b, k, 2 * d});
    auto zero = training::fill(vulkanContext, b * k * 2 * d);
    zero->setInput(pooledGradient2, 0);
    add(zero);
    auto place = training::channelCopy(vulkanContext, b * k, 1, {d, 0}, {2 * d, 0}, d);
    place->setInput(meansGradient, 0);
    place->setInput(pooledGradient2, 1);
    add(place);
    auto statisticsGradient2 = tensor({b, 2 * d, size, size});
    auto spread =
        training::objectBroadcast(vulkanContext, {b, 2 * d, size, size}, k, {s, s, step, step / 2}, 2 * d, 0, true);
    spread->setInput(pooledGradient2, 0);
    spread->setInput(objects, 1);
    spread->setInput(counts2, 2);
    spread->setInput(statisticsGradient2, 3);
    add(spread);
    auto outputGradient2 = backwardEncoderInput(checkEncoderLayers, statisticsGradient2);
    // The permuted latents are targets, so their gradient ends here.
    backwardSecondPass(checkBlocks, x2, logits2, outputGradient2, coordinates, classes, mlp ? &checkMlp : nullptr);
}

void PhotoEnhancerTraining::backwardSecondPass(std::vector<Block>& passBlocks, std::shared_ptr<TensorElement<float>> x,
                                               std::shared_ptr<TensorElement<float>> passLogits,
                                               std::shared_ptr<TensorElement<float>> passOutputGradient,
                                               std::shared_ptr<TensorElement<float>> passCoordinates,
                                               std::shared_ptr<TensorElement<float>> passClasses, MlpPass* passMlp) {
    const uint32_t b = config.batch, s = config.size;
    std::vector<std::string> redirected;
    for (const auto& parameter : parameters)
        if (parameter.name.rfind("generator.", 0) == 0) {
            gradientRedirect[parameter.name] = tensor({uint32_t(parameter.gradient->getDataElementCount())});
            redirected.push_back(parameter.name);
        }
    backwardGenerator(passBlocks, x, passLogits, passOutputGradient, tensor({b, 3, s, s}));
    // The texture's gradient adds atomically to the first pass's.
    if (config.texture.levels > 0) {
        auto texture = hashTextureBackward(vulkanContext, config.texture);
        texture->setInput(passCoordinates, 0);
        texture->setInput(passClasses, 1);
        texture->setInput(passBlocks[0].dIn, 2);
        texture->setInput(find("texture.table").gradient, 3);
        add(texture);
    }
    // The MLP texture's gradients (redirected as the generator's); the latents' end here. The feature texture's
    // tables' gradients add atomically to the first pass's.
    if (passMlp)
        backwardTextureMlp(*passMlp, passBlocks[0].dIn);
    if (config.textureBasis.enabled() && passCoordinates == coordinates2)
        backwardTextureBasis(passCoordinates, passClasses, latentImage2, passBlocks[0].dIn,
                             tensor({b, config.latentSize, s, s}));
    for (const auto& name : redirected) {
        const uint32_t count = uint32_t(find(name).gradient->getDataElementCount());
        auto accumulate = training::channelCopy(vulkanContext, 1, count, {1, 0}, {1, 0}, 1, true);
        accumulate->setInput(gradientRedirect[name], 0);
        accumulate->setInput(find(name).gradient, 1);
        add(accumulate);
    }
    gradientRedirect.clear();
}

void PhotoEnhancerTraining::buildTemporal() {
    const uint32_t b = config.batch, s = config.size, k = config.latentObjects, d = config.latentSize;
    const uint32_t gbuffer = config.gbufferChannels, features = textureFeatures();
    const uint32_t channelsIn = gbuffer + features + d;
    // The second views' G-buffers, texture features and latent G-buffer (the same latents).
    channels2 = tensor({b, gbuffer, s, s});
    coordinates2 = tensor({b, 3, s, s});
    classes2 = tensor({b, s, s});
    objects2 = tensor({b, s, s});
    flow = tensor({b, 2, s, s});
    flowValid = tensor({b, s, s});
    auto input2 = tensor({b, channelsIn, s, s});
    auto copy = training::channelCopy(vulkanContext, b, s * s, {gbuffer, 0}, {channelsIn, 0}, gbuffer);
    copy->setInput(channels2, 0);
    copy->setInput(input2, 1);
    add(copy);
    if (config.texture.levels > 0) {
        auto texture = hashTexture(vulkanContext, config.texture);
        texture->setInput(coordinates2, 0);
        texture->setInput(classes2, 1);
        texture->setInput(find("texture.table").value, 2);
        texture->setInput(input2, 3);
        add(texture);
    }
    auto splat =
        training::objectBroadcast(vulkanContext, {b, d, s, s}, k, {s, s, 1, 0}, channelsIn, gbuffer + features);
    splat->setInput(latentValues, 0);
    splat->setInput(objects2, 1);
    splat->setInput(counts, 2);
    splat->setInput(input2, 3);
    add(splat);
    const bool mlp = config.textureMlp.frequencies > 0;
    if (mlp)
        temporalMlp = forwardTextureMlp(coordinates2, classes2, input2);
    if (config.textureBasis.enabled()) {
        latentImage2 = tensor({b, d, s, s});
        auto image = training::objectBroadcast(vulkanContext, {b, d, s, s}, k, {s, s, 1, 0}, d, 0);
        image->setInput(latentValues, 0);
        image->setInput(objects2, 1);
        image->setInput(counts, 2);
        image->setInput(latentImage2, 3);
        add(image);
        forwardTextureBasis(coordinates2, classes2, latentImage2, input2);
    }
    if (!config.latentToDecoder)
        hideLatents(input2);
    temporalBlocks = blocks;
    auto [x2, logits2, output2] = forwardGenerator(temporalBlocks, input2);
    // The first render warped onto the second views against their renders.
    auto outputGradient2 = tensor({b, 3, s, s}), shares = tensor({b, 3, s, s});
    auto loss = training::flowWarpLoss(vulkanContext, {b, 3, s, s}, config.temporal);
    loss->setInput(output, 0);
    loss->setInput(output2, 1);
    loss->setInput(flow, 2);
    loss->setInput(flowValid, 3);
    loss->setInput(outputGradient2, 4);
    loss->setInput(shares, 5);
    add(loss);
    temporalTotal = tensor({1});
    auto total = training::sum(vulkanContext, b * 3 * s * s);
    total->setInput(shares, 0);
    total->setInput(temporalTotal, 1);
    add(total);
    // The latents' gradient from the second pass ends here.
    backwardSecondPass(temporalBlocks, x2, logits2, outputGradient2, coordinates2, classes2,
                       mlp ? &temporalMlp : nullptr);
}

void PhotoEnhancerTraining::hideLatents(std::shared_ptr<TensorElement<float>> generatorInput) {
    const uint32_t b = config.batch, s = config.size, d = config.latentSize;
    const uint32_t channelsIn = config.gbufferChannels + textureFeatures() + d;
    if (!latentZeros) {
        latentZeros = tensor({b, d, s, s});
        auto zero = training::fill(vulkanContext, b * d * s * s);
        zero->setInput(latentZeros, 0);
        add(zero);
    }
    auto clear = training::channelCopy(vulkanContext, b, s * s, {d, 0}, {channelsIn, channelsIn - d}, d);
    clear->setInput(latentZeros, 0);
    clear->setInput(generatorInput, 1);
    add(clear);
}

training::TextureBasisShape PhotoEnhancerTraining::basisShape(size_t level, bool accumulate) const {
    const auto& t = config.textureBasis;
    training::TextureBasisShape shape;
    shape.batch = config.batch;
    shape.pixels = config.size * config.size;
    shape.faces = t.faces;
    shape.layers = config.latentSize + 1;
    shape.channels = t.channels;
    shape.resolution = t.resolutions[level];
    shape.classes = config.texture.classes;
    shape.outputChannels = config.gbufferChannels + t.channels + config.latentSize;
    shape.outputOffset = config.gbufferChannels;
    shape.accumulate = accumulate;
    return shape;
}

void PhotoEnhancerTraining::forwardTextureBasis(std::shared_ptr<TensorElement<float>> passCoordinates,
                                                std::shared_ptr<TensorElement<float>> passClasses,
                                                std::shared_ptr<TensorElement<float>> passLatents,
                                                std::shared_ptr<TensorElement<float>> generatorInput) {
    for (size_t level = 0; level < config.textureBasis.resolutions.size(); ++level) {
        auto texture = training::textureBasis(vulkanContext, basisShape(level, level > 0));
        texture->setInput(passCoordinates, 0);
        texture->setInput(passClasses, 1);
        texture->setInput(find("texture.basis." + std::to_string(level)).value, 2);
        texture->setInput(passLatents, 3);
        texture->setInput(generatorInput, 4);
        add(texture);
    }
}

void PhotoEnhancerTraining::backwardTextureBasis(std::shared_ptr<TensorElement<float>> passCoordinates,
                                                 std::shared_ptr<TensorElement<float>> passClasses,
                                                 std::shared_ptr<TensorElement<float>> passLatents,
                                                 std::shared_ptr<TensorElement<float>> generatorInputGradient,
                                                 std::shared_ptr<TensorElement<float>> latentGradient) {
    for (size_t level = 0; level < config.textureBasis.resolutions.size(); ++level) {
        const auto& table = find("texture.basis." + std::to_string(level));
        auto texture = training::textureBasisBackward(vulkanContext, basisShape(level, level > 0));
        texture->setInput(passCoordinates, 0);
        texture->setInput(passClasses, 1);
        texture->setInput(table.value, 2);
        texture->setInput(passLatents, 3);
        texture->setInput(generatorInputGradient, 4);
        texture->setInput(table.gradient, 5);
        texture->setInput(latentGradient, 6);
        add(texture);
    }
}

void PhotoEnhancerTraining::buildLatentBackward() {
    const uint32_t b = config.batch, s = config.size, k = config.latentObjects, d = config.latentSize;
    const uint32_t gbuffer = config.gbufferChannels, features = textureFeatures();
    const uint32_t channelsIn = gbuffer + features + d;
    // The latent G-buffer's gradient, summed per object, through the VAE to the per-object statistics, spread evenly
    // over each object's pixels of the head's output.
    latentGBufferGradient = tensor({b, d, s, s});
    if (config.latentToDecoder) {
        auto slice = training::channelCopy(vulkanContext, b, s * s, {channelsIn, gbuffer + features}, {d, 0}, d);
        slice->setInput(blocks[0].dIn, 0);
        slice->setInput(latentGBufferGradient, 1);
        add(slice);
    } else {
        // The generator never saw the latents: only the texture's input gradient reaches them.
        auto zero = training::fill(vulkanContext, b * d * s * s);
        zero->setInput(latentGBufferGradient, 0);
        add(zero);
    }
    // The latents also feed the feature texture and the MLP texture: their latents' gradients add to the latent
    // G-buffer's.
    if (basisLatentGradient) {
        auto add = training::channelCopy(vulkanContext, b, s * s, {d, 0}, {d, 0}, d, true);
        add->setInput(basisLatentGradient, 0);
        add->setInput(latentGBufferGradient, 1);
        this->add(add);
    }
    if (mlpInputGradient) {
        const uint32_t encoding = 3 * (2 * config.textureMlp.frequencies + 1);
        auto add = training::channelCopy(vulkanContext, b, s * s, {encoding + d, encoding}, {d, 0}, d, true);
        add->setInput(mlpInputGradient, 0);
        add->setInput(latentGBufferGradient, 1);
        this->add(add);
    }
    latentGradient = tensor({b, k, d});
    latentSums = tensor({b, k});
    auto gather = training::objectMean(vulkanContext, {b, d, s, s}, k, {s, s, 1, 0}, false);
    gather->setInput(latentGBufferGradient, 0);
    gather->setInput(objects, 1);
    gather->setInput(latentGradient, 2);
    gather->setInput(latentSums, 3);
    add(gather);
    if (config.latentCodes > 0) {
        const auto& latentTable = find("latents.weight");
        auto scatter = training::embeddingBackward(vulkanContext, b, config.latentCodes, k * d, config.latentCodeDecay);
        scatter->setInput(latentTable.value, 0);
        scatter->setInput(rows, 1);
        scatter->setInput(latentGradient, 2);
        scatter->setInput(latentTable.gradient, 3);
        add(scatter);
        return;
    }
    pooledGradient = tensor({b, k, 2 * d});
    auto vae = training::vaeBackward(vulkanContext, b, k, d, config.klWeight, config.freeBits);
    vae->setInput(pooled, 0);
    vae->setInput(latentNoise, 1);
    vae->setInput(counts, 2);
    vae->setInput(latentGradient, 3);
    vae->setInput(pooledGradient, 4);
    add(vae);
    const uint32_t size = s >> encoderLayers.size(), step = s / size;
    statisticsGradient = tensor({b, 2 * d, size, size});
    auto spread =
        training::objectBroadcast(vulkanContext, {b, 2 * d, size, size}, k, {s, s, step, step / 2}, 2 * d, 0, true);
    spread->setInput(pooledGradient, 0);
    spread->setInput(objects, 1);
    spread->setInput(counts, 2);
    spread->setInput(statisticsGradient, 3);
    add(spread);
    // The head and the layers, down to the first convolution's parameters.
    const auto backwardConv = [&](const std::string& name, std::shared_ptr<TensorElement<float>> x, uint32_t channelsIn,
                                  uint32_t sizeIn, uint32_t channelsOut, uint32_t kernel, uint32_t stride,
                                  std::shared_ptr<TensorElement<float>> gradient,
                                  std::shared_ptr<TensorElement<float>> result) {
        const auto attributes = convAttributes(kernel, stride);
        const uint32_t sizeOut = sizeIn / stride;
        const std::vector<uint32_t> inShape{b, channelsIn, sizeIn, sizeIn},
            weightShape{channelsOut, channelsIn, kernel, kernel}, outShape{b, channelsOut, sizeOut, sizeOut};
        auto weights = training::convBackwardWeights(vulkanContext, attributes, inShape, weightShape, outShape);
        weights->setInput(x, 0);
        weights->setInput(gradient, 1);
        weights->setInput(find(name + "weight").gradient, 2);
        weights->setInput(find(name + "bias").gradient, 3);
        add(weights);
        if (!result)
            return;
        auto inputs = training::convBackwardInput(vulkanContext, attributes, inShape, weightShape, outShape);
        inputs->setInput(gradient, 0);
        inputs->setInput(find(name + "weight").value, 1);
        inputs->setInput(result, 2);
        add(inputs);
    };
    auto& last = encoderLayers.back();
    last.de = tensor({b, last.outputChannels, size, size});
    backwardConv("encoder.head.", last.e, last.outputChannels, size, 2 * d, 1, 1, statisticsGradient, last.de);
    for (int i = int(encoderLayers.size()) - 1; i >= 0; --i) {
        auto& layer = encoderLayers[size_t(i)];
        const uint32_t next = layer.inputSize / 2;
        const std::vector<uint32_t> shape{b, layer.outputChannels, next, next};
        layer.dn = tensor(shape);
        layer.da = tensor(shape);
        auto elu = training::activationBackward(vulkanContext, training::Activation::Elu, shape);
        elu->setInput(layer.n, 0);
        elu->setInput(layer.de, 1);
        elu->setInput(layer.dn, 2);
        add(elu);
        const std::string norm = "encoder.body." + std::to_string(3 * i + 1) + ".";
        auto inputs = training::channelNormBackwardInput(vulkanContext, shape);
        inputs->setInput(layer.a, 0);
        inputs->setInput(find(norm + "weight").value, 1);
        inputs->setInput(layer.stats, 2);
        inputs->setInput(layer.dn, 3);
        inputs->setInput(layer.da, 4);
        add(inputs);
        auto params = training::channelNormBackwardParameters(vulkanContext, shape);
        params->setInput(layer.a, 0);
        params->setInput(layer.stats, 1);
        params->setInput(layer.dn, 2);
        params->setInput(find(norm + "weight").gradient, 3);
        params->setInput(find(norm + "bias").gradient, 4);
        add(params);
        std::shared_ptr<TensorElement<float>> previous;
        if (i > 0) {
            auto& before = encoderLayers[size_t(i) - 1];
            before.de = tensor({b, before.outputChannels, layer.inputSize, layer.inputSize});
            previous = before.de;
        }
        backwardConv("encoder.body." + std::to_string(3 * i) + ".", layer.in, layer.inputChannels, layer.inputSize,
                     layer.outputChannels, 3, 2, layer.da, previous);
    }
}

void PhotoEnhancerTraining::buildDiscriminator() {
    const uint32_t conditioning = criticConditioning(config);
    const uint32_t b = config.batch, s = config.size, c = conditioning + 3, pixels = s * s;
    // The pairs (the G-buffer channels if conditional, then the image) of the real photos and of the generator's
    // images, and both together (real ones first) augmented for the discriminator's step.
    auto realPairs = tensor({b, c, s, s}), fakePairs = tensor({b, c, s, s});
    auto allPairs = tensor({2 * b, c, s, s}), augmented = tensor({2 * b, c, s, s});
    augmentation = tensor({2 * b, 6});
    if (config.separateRealGBuffer)
        realChannels = tensor({b, config.gbufferChannels, s, s});
    for (const auto& [pairs, image] : {std::pair{realPairs, targets}, std::pair{fakePairs, output}}) {
        if (conditioning > 0) {
            auto gbuffers =
                training::channelCopy(vulkanContext, b, pixels, {config.gbufferChannels, 0}, {c, 0}, conditioning);
            gbuffers->setInput(pairs == realPairs && realChannels ? realChannels : channels, 0);
            gbuffers->setInput(pairs, 1);
            add(gbuffers);
        }
        auto colour = training::channelCopy(vulkanContext, b, pixels, {3, 0}, {c, conditioning}, 3);
        colour->setInput(image, 0);
        colour->setInput(pairs, 1);
        add(colour);
    }
    for (uint32_t half = 0; half < 2; ++half) {
        // A contiguous block: one "channel" of b x c x pixels values.
        auto copy = training::channelCopy(vulkanContext, 1, b * c * pixels, {1, 0}, {2, half}, 1);
        copy->setInput(half == 0 ? realPairs : fakePairs, 0);
        copy->setInput(allPairs, 1);
        add(copy);
    }
    auto augment = training::augmentPair(vulkanContext, {2 * b, c, s, s});
    augment->setInput(allPairs, 0);
    augment->setInput(augmentation, 1);
    augment->setInput(augmented, 2);
    add(augment);

    // The discriminator's step: softplus(-D(real)) + softplus(D(fake)) per scale, then Adam.
    critics.push_back(std::make_unique<Critic>());
    Critic& step = *critics.back();
    step.batch = 2 * b;
    forwardCritic(step, augmented);
    for (auto& scale : step.scales) {
        const uint32_t half = b * scale.sizes.back() * scale.sizes.back();
        auto shares = tensor({2 * half});
        scale.logitGradient = tensor({2 * b, 1, scale.sizes.back(), scale.sizes.back()});
        for (const auto& [offset, sign] : {std::pair{0u, -1.0f}, std::pair{half, 1.0f}}) {
            auto loss = training::softplusLoss(vulkanContext, offset, half, sign, 1.0f);
            loss->setInput(scale.logits, 0);
            loss->setInput(scale.logitGradient, 1);
            loss->setInput(shares, 2);
            add(loss);
        }
        discriminatorTotals.push_back(tensor({1}));
        auto total = training::sum(vulkanContext, 2 * half);
        total->setInput(shares, 0);
        total->setInput(discriminatorTotals.back(), 1);
        add(total);
    }
    backwardCritic(step, true, nullptr, nullptr);
    discriminatorCounter = vulkanContext.create<TensorElement<uint32_t>>(std::vector<uint32_t>{1});
    auto counter = training::stepCounter(vulkanContext);
    counter->setInput(discriminatorCounter, 0);
    add(counter);
    for (const auto& parameter : parameters)
        if (parameter.group == Discriminator) {
            auto update = training::adamStep(vulkanContext, uint32_t(parameter.value->getDataElementCount()),
                                             config.discriminatorAdam);
            update->setInput(parameter.value, 0);
            update->setInput(parameter.gradient, 1);
            update->setInput(parameter.firstMoment, 2);
            update->setInput(parameter.secondMoment, 3);
            update->setInput(discriminatorCounter, 4);
            add(update);
        }

    // The generator's adversarial loss and feature matching against the updated discriminator, on the pairs as they
    // are: its gradient joins the image's.
    // Feature matching compares with the real pairs' features; it needs real and generated images of the same views.
    const bool matching = config.featureMatching > 0.0f;
    Critic* real = nullptr;
    if (matching) {
        critics.push_back(std::make_unique<Critic>());
        real = critics.back().get();
        real->batch = b;
        forwardCritic(*real, realPairs);
    }
    // With augmentGeneratorStep the generated pairs are augmented as in the discriminator's step.
    auto judged = fakePairs;
    if (config.augmentGeneratorStep) {
        generatorAugmentation = tensor({b, 6});
        judged = tensor({b, c, s, s});
        auto augmentFake = training::augmentPair(vulkanContext, {b, c, s, s});
        augmentFake->setInput(fakePairs, 0);
        augmentFake->setInput(generatorAugmentation, 1);
        augmentFake->setInput(judged, 2);
        add(augmentFake);
    }
    critics.push_back(std::make_unique<Critic>());
    Critic& fake = *critics.back();
    fake.batch = b;
    forwardCritic(fake, judged);
    for (auto& scale : fake.scales) {
        const uint32_t count = b * scale.sizes.back() * scale.sizes.back();
        auto shares = tensor({count});
        scale.logitGradient = tensor({b, 1, scale.sizes.back(), scale.sizes.back()});
        auto loss = training::softplusLoss(vulkanContext, 0, count, -1.0f, config.adversarial);
        loss->setInput(scale.logits, 0);
        loss->setInput(scale.logitGradient, 1);
        loss->setInput(shares, 2);
        add(loss);
        adversarialTotals.push_back(tensor({1}));
        auto total = training::sum(vulkanContext, count);
        total->setInput(shares, 0);
        total->setInput(adversarialTotals.back(), 1);
        add(total);
    }
    auto pairGradient = tensor({b, c, s, s});
    const auto featureGradient = [&](Critic& critic, size_t scaleIndex, size_t layer) {
        auto& scale = critic.scales[scaleIndex];
        const uint32_t count = uint32_t(scale.activated[layer]->getDataElementCount());
        auto shares = tensor({count});
        auto distance = training::l1Distance(vulkanContext, count, config.featureMatching / 4.0f, true);
        distance->setInput(scale.activated[layer], 0);
        distance->setInput(real->scales[scaleIndex].activated[layer], 1);
        distance->setInput(scale.activatedGradient[layer], 2);
        distance->setInput(shares, 3);
        add(distance);
        matchingTotals.push_back(tensor({1}));
        auto total = training::sum(vulkanContext, count);
        total->setInput(shares, 0);
        total->setInput(matchingTotals.back(), 1);
        add(total);
    };
    backwardCritic(fake, false, pairGradient,
                   matching ? std::function<void(Critic&, size_t, size_t)>(featureGradient)
                            : std::function<void(Critic&, size_t, size_t)>());
    if (config.augmentGeneratorStep) {
        auto unaugmented = tensor({b, c, s, s});
        auto back = training::augmentPairBackward(vulkanContext, {b, c, s, s});
        back->setInput(generatorAugmentation, 0);
        back->setInput(pairGradient, 1);
        back->setInput(unaugmented, 2);
        add(back);
        pairGradient = unaugmented;
    }
    auto colour = training::channelCopy(vulkanContext, b, pixels, {c, conditioning}, {3, 0}, 3, true);
    colour->setInput(pairGradient, 0);
    colour->setInput(outputGradient, 1);
    add(colour);
}

void PhotoEnhancerTraining::forwardCritic(Critic& critic, std::shared_ptr<TensorElement<float>> pairs) {
    const uint32_t n = critic.batch, c = criticConditioning(config) + 3;
    for (uint32_t scaleIndex = 0; scaleIndex < config.criticScales; ++scaleIndex) {
        Critic::Scale scale;
        scale.size = config.size >> scaleIndex;
        if (scaleIndex == 0) {
            scale.input = pairs;
        } else {
            // The previous scale's input, average-pooled.
            scale.input = tensor({n, c, scale.size, scale.size});
            const uint32_t previous = scale.size * 2;
            auto pool = training::averagePool(vulkanContext, {n, c, previous, previous});
            pool->setInput(critic.scales.back().input, 0);
            pool->setInput(scale.input, 1);
            add(pool);
        }
        scale.scaled = tensor({n, c, scale.size, scale.size});
        auto affine = training::affine(vulkanContext, n * c * scale.size * scale.size, 2.0f, -1.0f);
        affine->setInput(scale.input, 0);
        affine->setInput(scale.scaled, 1);
        add(affine);
        const std::string prefix = "discriminator.discriminators." + std::to_string(scaleIndex) + ".";
        auto x = scale.scaled;
        uint32_t channels = c, size = scale.size;
        for (uint32_t layer = 0; layer <= 4; ++layer) {
            const bool logits = layer == 4;
            const uint32_t width = logits ? 1 : kCriticWidths[layer], stride = logits ? 1 : kCriticStrides[layer];
            const uint32_t next = stride == 2 ? size / 2 : size - 1;
            layers::ConvAttributes attributes;
            attributes.kernelShape = {4, 4};
            attributes.strides = {stride, stride};
            attributes.pads = {1, 1, 1, 1};
            const std::string name = logits ? prefix + "logits." : prefix + "layers." + std::to_string(layer) + ".0.";
            auto y = tensor({n, width, next, next});
            auto conv = layers::conv(vulkanContext, attributes, {n, channels, size, size}, {width, channels, 4, 4},
                                     {width}, {n, width, next, next});
            conv->setInput(x, 0);
            conv->setInput(find(name + "weight").value, 1);
            conv->setInput(find(name + "bias").value, 2);
            conv->setInput(y, 3);
            add(conv);
            scale.sizes.push_back(next);
            if (logits) {
                scale.logits = y;
                break;
            }
            auto activated = tensor({n, width, next, next});
            auto relu = training::leakyRelu(vulkanContext, {n, width, next, next});
            relu->setInput(y, 0);
            relu->setInput(activated, 1);
            add(relu);
            scale.pre.push_back(y);
            scale.activated.push_back(activated);
            x = activated;
            channels = width;
            size = next;
        }
        critic.scales.push_back(scale);
    }
}

void PhotoEnhancerTraining::backwardCritic(Critic& critic, bool withWeights,
                                           std::shared_ptr<TensorElement<float>> inputGradient,
                                           const std::function<void(Critic&, size_t, size_t)>& featureGradient) {
    const uint32_t n = critic.batch, c = criticConditioning(config) + 3;
    for (size_t scaleIndex = 0; scaleIndex < critic.scales.size(); ++scaleIndex) {
        auto& scale = critic.scales[scaleIndex];
        const std::string prefix = "discriminator.discriminators." + std::to_string(scaleIndex) + ".";
        scale.preGradient.resize(4);
        scale.activatedGradient.resize(4);
        auto gradient = scale.logitGradient;
        for (int layer = 4; layer >= 0; --layer) {
            const bool logits = layer == 4;
            const uint32_t width = logits ? 1 : kCriticWidths[layer];
            const uint32_t channels = layer == 0 ? c : kCriticWidths[layer - 1];
            const uint32_t size = layer == 0 ? scale.size : scale.sizes[size_t(layer) - 1];
            const uint32_t next = scale.sizes[size_t(layer)];
            const std::vector<uint32_t> outShape{n, width, next, next};
            if (!logits) {
                // The leaky ReLU's input gradient from its output's.
                scale.preGradient[size_t(layer)] = tensor(outShape);
                auto relu = training::activationBackward(vulkanContext, training::Activation::LeakyRelu, outShape);
                relu->setInput(scale.pre[size_t(layer)], 0);
                relu->setInput(scale.activatedGradient[size_t(layer)], 1);
                relu->setInput(scale.preGradient[size_t(layer)], 2);
                add(relu);
                gradient = scale.preGradient[size_t(layer)];
            }
            layers::ConvAttributes attributes;
            attributes.kernelShape = {4, 4};
            attributes.strides = {logits ? 1u : kCriticStrides[layer], logits ? 1u : kCriticStrides[layer]};
            attributes.pads = {1, 1, 1, 1};
            const std::string name = logits ? prefix + "logits." : prefix + "layers." + std::to_string(layer) + ".0.";
            const std::vector<uint32_t> inShape{n, channels, size, size}, weightShape{width, channels, 4, 4};
            auto input = layer == 0 ? scale.scaled : scale.activated[size_t(layer) - 1];
            if (withWeights) {
                auto weights = training::convBackwardWeights(vulkanContext, attributes, inShape, weightShape, outShape);
                weights->setInput(input, 0);
                weights->setInput(gradient, 1);
                weights->setInput(find(name + "weight").gradient, 2);
                weights->setInput(find(name + "bias").gradient, 3);
                add(weights);
            }
            if (layer == 0 && !inputGradient)
                break;
            auto result = tensor(inShape);
            auto inputs = training::convBackwardInput(vulkanContext, attributes, inShape, weightShape, outShape);
            inputs->setInput(gradient, 0);
            inputs->setInput(find(name + "weight").value, 1);
            inputs->setInput(result, 2);
            add(inputs);
            if (layer > 0) {
                scale.activatedGradient[size_t(layer) - 1] = result;
                if (featureGradient)
                    featureGradient(critic, scaleIndex, size_t(layer) - 1);
            } else {
                scale.scaledGradient = result;
            }
        }
        if (!inputGradient)
            continue;
        // Through the input's scaling (times 2) and, for the second scale, its average pool, into the pairs' gradient.
        const uint32_t count = n * c * scale.size * scale.size;
        if (scaleIndex == 0) {
            auto affine = training::affine(vulkanContext, count, 2.0f, 0.0f, true);
            affine->setInput(scale.scaledGradient, 0);
            affine->setInput(inputGradient, 1);
            add(affine);
        } else {
            scale.inputGradient = tensor({n, c, scale.size, scale.size});
            auto affine = training::affine(vulkanContext, count, 2.0f, 0.0f, true);
            affine->setInput(scale.scaledGradient, 0);
            affine->setInput(scale.inputGradient, 1);
            add(affine);
            // Back through the average pools, one per scale, into the full-resolution pairs' gradient.
            auto gradient = scale.inputGradient;
            for (uint32_t size = scale.size * 2; size <= config.size; size *= 2) {
                const bool last = size == config.size;
                auto larger = last ? inputGradient : tensor({n, c, size, size});
                auto pool = training::averagePoolBackward(vulkanContext, {n, c, size, size}, last);
                pool->setInput(gradient, 0);
                pool->setInput(larger, 1);
                add(pool);
                gradient = larger;
            }
        }
    }
}

std::vector<float> PhotoEnhancerTraining::randomAugmentation() {
    // As train_photo_enhancer.augment_pair(), once for the real and once for the fake pairs: brightness and contrast
    // per sample, one roll and one cutout per call.
    const int s = int(config.size);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    std::vector<float> numbers;
    for (int half = 0; half < 2; ++half) {
        std::uniform_int_distribution<int> shift(-s / 8, s / 8), position(0, s - 1);
        const int shiftY = shift(random), shiftX = shift(random);
        // Python's floor division of the first shift by two.
        const int rollY = shiftY >= 0 ? shiftY / 2 : -((-shiftY + 1) / 2);
        const int cy = position(random), cx = position(random);
        for (uint32_t k = 0; k < config.batch; ++k) {
            const float brightness = (uniform(random) - 0.5f) * 0.4f, contrast = uniform(random) + 0.5f;
            numbers.insert(numbers.end(), {brightness, contrast, float(rollY), float(shiftX), float(cy), float(cx)});
        }
    }
    return numbers;
}

void PhotoEnhancerTraining::setObjects(const std::vector<float>& values) {
    if (!objects)
        throw std::logic_error("PhotoEnhancerTraining: objects need latentSize > 0");
    objects->getDataBuffer(0).memcopyFrom(values);
}

void PhotoEnhancerTraining::setLatents(const std::vector<float>& values) {
    if (!latentValues || latentNoise)
        throw std::logic_error("PhotoEnhancerTraining: latents are given only forward only without the encoder");
    latentValues->getDataBuffer(0).memcopyFrom(values);
}

void PhotoEnhancerTraining::setRealGBuffer(const std::vector<float>& values) {
    if (!realChannels)
        throw std::logic_error(
            "PhotoEnhancerTraining: a real G-buffer needs separateRealGBuffer and the discriminator");
    realChannels->getDataBuffer(0).memcopyFrom(values);
}

void PhotoEnhancerTraining::setLatentNoise(const std::vector<float>& noise) {
    if (!latentNoise)
        throw std::logic_error("PhotoEnhancerTraining: latent noise needs the encoder");
    latentNoise->getDataBuffer(0).memcopyFrom(noise);
    latentNoiseGiven = true;
}

void PhotoEnhancerTraining::setLatentPermutation(const std::vector<float>& permutation) {
    if (!latentPermutation)
        throw std::logic_error("PhotoEnhancerTraining: a latent permutation needs the latent check");
    latentPermutation->getDataBuffer(0).memcopyFrom(permutation);
    latentPermutationGiven = true;
}

void PhotoEnhancerTraining::setSecondView(const std::vector<float>& channelValues,
                                          const std::vector<float>& coordinateValues,
                                          const std::vector<float>& classValues, const std::vector<float>& objectValues,
                                          const std::vector<float>& flowValues, const std::vector<float>& validValues) {
    if (!flow)
        throw std::logic_error("PhotoEnhancerTraining: second views need the temporal loss");
    channels2->getDataBuffer(0).memcopyFrom(channelValues);
    if (config.texture.levels > 0 || config.textureMlp.frequencies > 0 || config.textureBasis.enabled()) {
        coordinates2->getDataBuffer(0).memcopyFrom(coordinateValues);
        classes2->getDataBuffer(0).memcopyFrom(classValues);
    }
    objects2->getDataBuffer(0).memcopyFrom(objectValues);
    flow->getDataBuffer(0).memcopyFrom(flowValues);
    flowValid->getDataBuffer(0).memcopyFrom(validValues);
}

float PhotoEnhancerTraining::temporalLoss() const {
    if (!temporalTotal)
        return 0.0f;
    std::vector<float> value(1);
    temporalTotal->getDataBuffer(0).memcopyTo(value);
    return value[0] / config.temporal;
}

float PhotoEnhancerTraining::latentCheckLoss() const {
    if (!checkTotal)
        return 0.0f;
    std::vector<float> value(1);
    checkTotal->getDataBuffer(0).memcopyTo(value);
    return value[0] / config.latentCheck;
}

std::vector<float> PhotoEnhancerTraining::latents() const {
    std::vector<float> values(latentValues ? latentValues->getDataElementCount() : 0);
    if (latentValues)
        latentValues->getDataBuffer(0).memcopyTo(values);
    return values;
}

float PhotoEnhancerTraining::klLoss() const {
    if (!klTotal || config.klWeight <= 0.0f)
        return 0.0f;
    std::vector<float> value(1);
    klTotal->getDataBuffer(0).memcopyTo(value);
    return value[0] / config.klWeight;
}

void PhotoEnhancerTraining::setAugmentation(const std::vector<float>& numbers) {
    if (!augmentation || numbers.size() != 12 * config.batch)
        throw std::invalid_argument("PhotoEnhancerTraining: augmentation needs the discriminator and 2 x batch x 6");
    augmentation->getDataBuffer(0).memcopyFrom(numbers);
    augmentationGiven = true;
}

float PhotoEnhancerTraining::discriminatorLoss() const {
    float sum = 0.0f;
    for (const auto& total : discriminatorTotals) {
        std::vector<float> value(1);
        total->getDataBuffer(0).memcopyTo(value);
        sum += value[0];
    }
    return sum;
}

float PhotoEnhancerTraining::adversarialLoss() const {
    float sum = 0.0f;
    for (const auto& total : adversarialTotals) {
        std::vector<float> value(1);
        total->getDataBuffer(0).memcopyTo(value);
        sum += value[0];
    }
    return config.adversarial > 0.0f ? sum / config.adversarial : 0.0f;
}

float PhotoEnhancerTraining::featureMatchingLoss() const {
    float sum = 0.0f;
    for (const auto& total : matchingTotals) {
        std::vector<float> value(1);
        total->getDataBuffer(0).memcopyTo(value);
        sum += value[0];
    }
    return config.featureMatching > 0.0f ? sum / config.featureMatching : 0.0f;
}

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

std::tuple<std::shared_ptr<TensorElement<float>>, std::shared_ptr<TensorElement<float>>,
           std::shared_ptr<TensorElement<float>>>
PhotoEnhancerTraining::forwardGenerator(std::vector<Block>& blocks, std::shared_ptr<TensorElement<float>> input) {
    const uint32_t b = config.batch, s = config.size, levels = uint32_t(config.widths.size());
    const auto& w = config.widths;
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
    const uint32_t outputs = 3;
    auto logits = tensor({b, outputs, s, s}), output = tensor({b, 3, s, s});
    {
        auto conv = layers::conv(vulkanContext, convAttributes(1, 1), {b, w[0], s, s}, {outputs, w[0], 1, 1}, {outputs},
                                 {b, outputs, s, s});
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
    return {x, logits, output};
}

void PhotoEnhancerTraining::backwardGenerator(std::vector<Block>& blocks, std::shared_ptr<TensorElement<float>> x,
                                              std::shared_ptr<TensorElement<float>> logits,
                                              std::shared_ptr<TensorElement<float>> outputGradient,
                                              std::shared_ptr<TensorElement<float>> logitGradient) {
    const uint32_t b = config.batch, s = config.size, levels = uint32_t(config.widths.size());
    const auto& w = config.widths;
    std::shared_ptr<TensorElement<float>> dx = tensor({b, w[0], s, s});
    const uint32_t outputs = 3;
    {
        auto sigmoid = training::activationBackward(vulkanContext, training::Activation::Sigmoid, {b, 3, s, s});
        sigmoid->setInput(logits, 0);
        sigmoid->setInput(outputGradient, 1);
        sigmoid->setInput(logitGradient, 2);
        add(sigmoid);
        const auto attributes = convAttributes(1, 1);
        auto weights = training::convBackwardWeights(vulkanContext, attributes, {b, w[0], s, s}, {outputs, w[0], 1, 1},
                                                     {b, outputs, s, s});
        weights->setInput(x, 0);
        weights->setInput(logitGradient, 1);
        weights->setInput(gradientOf("generator.out.weight"), 2);
        weights->setInput(gradientOf("generator.out.bias"), 3);
        add(weights);
        auto inputs = training::convBackwardInput(vulkanContext, attributes, {b, w[0], s, s}, {outputs, w[0], 1, 1},
                                                  {b, outputs, s, s});
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
}

std::shared_ptr<TensorElement<float>> PhotoEnhancerTraining::gradientOf(const std::string& name) const {
    const auto redirected = gradientRedirect.find(name);
    return redirected != gradientRedirect.end() ? redirected->second : find(name).gradient;
}

uint32_t savedLatentSize(const std::string& directory) {
    std::ifstream file(std::filesystem::path(directory) / "texture.txt");
    for (std::string line; std::getline(file, line);) {
        std::istringstream words(line);
        std::string key;
        uint32_t size = 0;
        if (words >> key >> size && key == "latent")
            return size;
    }
    const auto head = std::filesystem::path(directory) / "encoder.head.bias.bin";
    return std::filesystem::exists(head) ? uint32_t(std::filesystem::file_size(head) / sizeof(float) / 2) : 0;
}

TextureBasis TextureBasis::fromFile(const std::string& path) {
    TextureBasis basis;
    std::ifstream file(path);
    for (std::string line; std::getline(file, line);) {
        std::istringstream words(line);
        std::string key;
        if (words >> key && key == "basis") {
            words >> basis.channels >> basis.faces;
            for (uint32_t resolution; words >> resolution;)
                basis.resolutions.push_back(resolution);
        } else if (key == "basis-regularize")
            words >> basis.decay >> basis.smoothness;
    }
    return basis;
}

bool savedLatentToDecoder(const std::string& directory) {
    std::ifstream file(std::filesystem::path(directory) / "texture.txt");
    for (std::string line; std::getline(file, line);) {
        std::istringstream words(line);
        std::string key;
        int value = 1;
        if (words >> key >> value && key == "decoder-latent")
            return value != 0;
    }
    return true;
}

TextureMlp TextureMlp::fromFile(const std::string& path) {
    TextureMlp mlp;
    std::ifstream file(path);
    for (std::string line; std::getline(file, line);) {
        std::istringstream words(line);
        std::string key;
        if (words >> key && key == "mlp")
            words >> mlp.frequencies >> mlp.width >> mlp.outputs >> mlp.hidden;
        else if (key == "mlp-residual")
            for (uint32_t frequencies, width; words >> frequencies >> width;)
                mlp.residual.emplace_back(frequencies, width);
    }
    return mlp;
}

uint32_t PhotoEnhancerTraining::textureFeatures() const {
    if (config.textureBasis.enabled())
        return config.textureBasis.channels;
    return config.textureMlp.frequencies > 0 ? config.textureMlp.outputs
                                             : config.texture.levels * config.texture.features;
}

PhotoEnhancerTraining::MlpPass PhotoEnhancerTraining::forwardMlp(const std::string& prefix, uint32_t frequencies,
                                                                 uint32_t width, uint32_t hidden, uint32_t outputs,
                                                                 std::shared_ptr<TensorElement<float>> passCoordinates,
                                                                 std::shared_ptr<TensorElement<float>> passClasses,
                                                                 std::shared_ptr<TensorElement<float>> generatorInput,
                                                                 uint32_t latentOffset) {
    const uint32_t b = config.batch, s = config.size, d = config.latentSize;
    const uint32_t encoding = 3 * (2 * frequencies + 1), channelsIn = generatorInput->getDimensions()[1];
    MlpPass pass;
    pass.encoding = tensor({b, encoding, s, s});
    auto encode = training::positionalEncoding(vulkanContext, b, s * s, frequencies, config.texture.classes);
    encode->setInput(passCoordinates, 0);
    encode->setInput(passClasses, 1);
    encode->setInput(pass.encoding, 2);
    add(encode);
    pass.in = tensor({b, encoding + d, s, s});
    auto first = training::channelCopy(vulkanContext, b, s * s, {encoding, 0}, {encoding + d, 0}, encoding);
    first->setInput(pass.encoding, 0);
    first->setInput(pass.in, 1);
    add(first);
    auto latents =
        training::channelCopy(vulkanContext, b, s * s, {channelsIn, latentOffset}, {encoding + d, encoding}, d);
    latents->setInput(generatorInput, 0);
    latents->setInput(pass.in, 1);
    add(latents);
    auto x = pass.in;
    uint32_t previous = encoding + d;
    for (uint32_t layer = 0; layer <= hidden; ++layer) {
        const uint32_t next = layer == hidden ? outputs : width;
        const std::string name = prefix + std::to_string(2 * layer) + ".";
        auto y = tensor({b, next, s, s});
        auto conv = layers::conv(vulkanContext, convAttributes(1, 1), {b, previous, s, s}, {next, previous, 1, 1},
                                 {next}, {b, next, s, s});
        conv->setInput(x, 0);
        conv->setInput(find(name + "weight").value, 1);
        conv->setInput(find(name + "bias").value, 2);
        conv->setInput(y, 3);
        add(conv);
        pass.pre.push_back(y);
        if (layer < hidden) {
            auto activated = tensor({b, next, s, s});
            auto elu = layers::elu(vulkanContext, {b, next, s, s}, 1.0f);
            elu->setInput(y, 0);
            elu->setInput(activated, 1);
            add(elu);
            pass.activated.push_back(activated);
            x = activated;
        }
        previous = next;
    }
    return pass;
}

std::shared_ptr<TensorElement<float>>
PhotoEnhancerTraining::backwardMlp(MlpPass& pass, const std::string& prefix, uint32_t frequencies, uint32_t width,
                                   uint32_t hidden, uint32_t outputs,
                                   std::shared_ptr<TensorElement<float>> outputGradient) {
    const uint32_t b = config.batch, s = config.size, d = config.latentSize;
    const uint32_t encoding = 3 * (2 * frequencies + 1);
    auto gradient = outputGradient;
    std::shared_ptr<TensorElement<float>> inputGradient;
    for (int layer = int(hidden); layer >= 0; --layer) {
        const uint32_t next = uint32_t(layer) == hidden ? outputs : width;
        const uint32_t previous = layer == 0 ? encoding + d : width;
        const std::string name = prefix + std::to_string(2 * layer) + ".";
        const auto attributes = convAttributes(1, 1);
        const std::vector<uint32_t> inShape{b, previous, s, s}, weightShape{next, previous, 1, 1},
            outShape{b, next, s, s};
        auto x = layer == 0 ? pass.in : pass.activated[size_t(layer) - 1];
        auto weights = training::convBackwardWeights(vulkanContext, attributes, inShape, weightShape, outShape);
        weights->setInput(x, 0);
        weights->setInput(gradient, 1);
        weights->setInput(gradientOf(name + "weight"), 2);
        weights->setInput(gradientOf(name + "bias"), 3);
        add(weights);
        auto result = tensor(inShape);
        auto inputs = training::convBackwardInput(vulkanContext, attributes, inShape, weightShape, outShape);
        inputs->setInput(gradient, 0);
        inputs->setInput(find(name + "weight").value, 1);
        inputs->setInput(result, 2);
        add(inputs);
        if (layer == 0) {
            inputGradient = result;
            break;
        }
        // Through the previous layer's ELU.
        gradient = tensor(inShape);
        auto elu = training::activationBackward(vulkanContext, training::Activation::Elu, inShape);
        elu->setInput(pass.pre[size_t(layer) - 1], 0);
        elu->setInput(result, 1);
        elu->setInput(gradient, 2);
        add(elu);
    }
    return inputGradient;
}

PhotoEnhancerTraining::MlpPass
PhotoEnhancerTraining::forwardTextureMlp(std::shared_ptr<TensorElement<float>> passCoordinates,
                                         std::shared_ptr<TensorElement<float>> passClasses,
                                         std::shared_ptr<TensorElement<float>> generatorInput) {
    const uint32_t b = config.batch, s = config.size, d = config.latentSize, gbuffer = config.gbufferChannels;
    const auto& t = config.textureMlp;
    const uint32_t channelsIn = gbuffer + t.outputs + d;
    auto pass = forwardMlp("generator.texture.", t.frequencies, t.width, t.hidden, t.outputs, passCoordinates,
                           passClasses, generatorInput, gbuffer + t.outputs);
    auto out = training::channelCopy(vulkanContext, b, s * s, {t.outputs, 0}, {channelsIn, gbuffer}, t.outputs);
    out->setInput(pass.pre.back(), 0);
    out->setInput(generatorInput, 1);
    add(out);
    // The residual blocks' features add to the first block's.
    for (size_t j = 0; j < t.residual.size(); ++j) {
        pass.residual.push_back(forwardMlp("generator.texture_residual." + std::to_string(j) + ".", t.residual[j].first,
                                           t.residual[j].second, t.hidden, t.outputs, passCoordinates, passClasses,
                                           generatorInput, gbuffer + t.outputs));
        auto sum =
            training::channelCopy(vulkanContext, b, s * s, {t.outputs, 0}, {channelsIn, gbuffer}, t.outputs, true);
        sum->setInput(pass.residual.back().pre.back(), 0);
        sum->setInput(generatorInput, 1);
        add(sum);
    }
    return pass;
}

std::shared_ptr<TensorElement<float>>
PhotoEnhancerTraining::backwardTextureMlp(MlpPass& pass, std::shared_ptr<TensorElement<float>> generatorInputGradient) {
    const uint32_t b = config.batch, s = config.size, d = config.latentSize, gbuffer = config.gbufferChannels;
    const auto& t = config.textureMlp;
    const uint32_t channelsIn = gbuffer + t.outputs + d;
    auto gradient = tensor({b, t.outputs, s, s});
    auto slice = training::channelCopy(vulkanContext, b, s * s, {channelsIn, gbuffer}, {t.outputs, 0}, t.outputs);
    slice->setInput(generatorInputGradient, 0);
    slice->setInput(gradient, 1);
    add(slice);
    auto inputGradient = backwardMlp(pass, "generator.texture.", t.frequencies, t.width, t.hidden, t.outputs, gradient);
    // Every residual block gets the same feature gradient; their latents' gradients add to the first block's.
    const uint32_t encoding = 3 * (2 * t.frequencies + 1);
    for (size_t j = 0; j < pass.residual.size(); ++j) {
        auto blockGradient = backwardMlp(pass.residual[j], "generator.texture_residual." + std::to_string(j) + ".",
                                         t.residual[j].first, t.residual[j].second, t.hidden, t.outputs, gradient);
        const uint32_t blockEncoding = 3 * (2 * t.residual[j].first + 1);
        auto latents = training::channelCopy(vulkanContext, b, s * s, {blockEncoding + d, blockEncoding},
                                             {encoding + d, encoding}, d, true);
        latents->setInput(blockGradient, 0);
        latents->setInput(inputGradient, 1);
        add(latents);
    }
    return inputGradient;
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
        params->setInput(gradientOf(name + ".weight"), 3);
        params->setInput(gradientOf(name + ".bias"), 4);
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
        weights->setInput(gradientOf(name + ".weight"), 2);
        weights->setInput(gradientOf(name + ".bias"), 3);
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
    // The coordinates and classes only feed the texture.
    if ((config.texture.levels > 0 || config.textureMlp.frequencies > 0 || config.textureBasis.enabled()) &&
        !config.textureOff) {
        coordinates->getDataBuffer(0).memcopyFrom(coordinateValues);
        classes->getDataBuffer(0).memcopyFrom(classValues);
    }
    // Forward only, the targets are not part of the graph.
    if (!config.forwardOnly || (config.latentSize > 0 && config.encodeLatents))
        targets->getDataBuffer(0).memcopyFrom(targetValues);
    if (config.appearances > 0 || (config.latentCodes > 0 && !config.forwardOnly))
        rows->getDataBuffer(0).memcopyFrom(rowValues);
}

void PhotoEnhancerTraining::step() {
    // The VAE's noise: random while training, none (the latents' means) when rendering from the encoder.
    if (latentNoise && !latentNoiseGiven) {
        std::vector<float> noise(latentNoise->getDataElementCount(), 0.0f);
        if (!config.forwardOnly) {
            std::normal_distribution<float> normal(0.0f, 1.0f);
            for (auto& v : noise)
                v = normal(random);
        }
        latentNoise->getDataBuffer(0).memcopyFrom(noise);
    }
    latentNoiseGiven = false;
    // The latent check renders view k with the latents of view k + shift (cyclically), shift 1 .. batch - 1.
    if (latentPermutation && !latentPermutationGiven) {
        const uint32_t b = config.batch;
        const uint32_t shift = b > 1 ? 1u + uint32_t(random() % (b - 1)) : 0u;
        std::vector<float> permutation(b);
        for (uint32_t i = 0; i < b; ++i)
            permutation[i] = float((i + shift) % b);
        latentPermutation->getDataBuffer(0).memcopyFrom(permutation);
    }
    latentPermutationGiven = false;
    if (augmentation && !augmentationGiven)
        augmentation->getDataBuffer(0).memcopyFrom(randomAugmentation());
    if (generatorAugmentation) {
        auto numbers = randomAugmentation();
        numbers.resize(config.batch * 6);
        generatorAugmentation->getDataBuffer(0).memcopyFrom(numbers);
    }
    augmentationGiven = false;
    graph->submitAndWait(vulkanContext.getGraphicsQueue(), 0);
}

void PhotoEnhancerTraining::save(const std::string& directory, bool averages) const {
    std::filesystem::create_directories(directory);
    // The texture's shape (HashTextureShape::fromFile()), so that the parameters can be loaded without knowing it.
    {
        const auto& shape = config.texture;
        std::ofstream texture(std::filesystem::path(directory) / "texture.txt");
        texture << "width " << shape.width << "\nheight " << shape.height << "\nlevels " << shape.levels
                << "\nfeatures " << shape.features << "\ntable " << shape.tableSize << "\nresolutions";
        for (uint32_t r : shape.resolutions)
            texture << " " << r;
        texture << "\nclasses";
        for (uint32_t c : shape.classes)
            texture << " " << c;
        texture << "\n";
        if (config.latentSize > 0)
            texture << "latent " << config.latentSize << "\n";
        if (config.latentSize > 0 && !config.latentToDecoder)
            texture << "decoder-latent 0\n";
        if (config.textureBasis.enabled()) {
            texture << "basis " << config.textureBasis.channels << " " << config.textureBasis.faces;
            for (uint32_t r : config.textureBasis.resolutions)
                texture << " " << r;
            texture << "\n";
            if (config.textureBasis.decay > 0.0f || config.textureBasis.smoothness > 0.0f)
                texture << "basis-regularize " << config.textureBasis.decay << " " << config.textureBasis.smoothness
                        << "\n";
        }
        const auto& t = config.textureMlp;
        if (t.frequencies > 0)
            texture << "mlp " << t.frequencies << " " << t.width << " " << t.outputs << " " << t.hidden << "\n";
        if (t.frequencies > 0 && !t.residual.empty()) {
            texture << "mlp-residual";
            for (const auto& [frequencies, width] : t.residual)
                texture << " " << frequencies << " " << width;
            texture << "\n";
        }
    }
    std::ofstream manifest(std::filesystem::path(directory) / "parameters.txt");
    for (const auto& parameter : parameters) {
        const auto& source =
            averages && !config.forwardOnly && parameter.group != Discriminator ? parameter.average : parameter.value;
        std::vector<float> values(source->getDataElementCount());
        source->getDataBuffer(0).memcopyTo(values);
        std::ofstream(std::filesystem::path(directory) / (parameter.name + ".bin"), std::ios::binary)
            .write(reinterpret_cast<const char*>(values.data()), std::streamsize(values.size() * sizeof(float)));
        manifest << parameter.name;
        for (uint32_t d : parameter.value->getDimensions())
            manifest << " " << d;
        manifest << "\n";
    }
}

void PhotoEnhancerTraining::load(const std::string& directory) {
    for (const auto& parameter : parameters) {
        const auto path = std::filesystem::path(directory) / (parameter.name + ".bin");
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file)
            throw std::runtime_error("PhotoEnhancerTraining: cannot read " + path.string());
        std::vector<float> values(size_t(file.tellg()) / sizeof(float));
        file.seekg(0);
        file.read(reinterpret_cast<char*>(values.data()), std::streamsize(values.size() * sizeof(float)));
        setParameter(parameter.name, values);
    }
}

std::vector<float> PhotoEnhancerTraining::meanCode() const {
    if (config.appearances == 0)
        return {};
    const auto& codes = find("codes.weight");
    std::vector<float> values(codes.value->getDataElementCount());
    (config.forwardOnly ? codes.value : codes.average)->getDataBuffer(0).memcopyTo(values);
    std::vector<float> mean(config.appearanceSize, 0.0f);
    for (size_t i = 0; i < values.size(); ++i)
        mean[i % config.appearanceSize] += values[i] / float(config.appearances);
    return mean;
}

void PhotoEnhancerTraining::setAppearance(const std::vector<float>& code) {
    if (!config.forwardOnly || code.size() != config.appearanceSize)
        throw std::invalid_argument("PhotoEnhancerTraining: setAppearance needs forward only and a code of its size");
    std::vector<float> table;
    for (uint32_t row = 0; row < config.appearances; ++row)
        table.insert(table.end(), code.begin(), code.end());
    find("codes.weight").value->getDataBuffer(0).memcopyFrom(table);
}

std::vector<std::pair<std::string, double>> PhotoEnhancerTraining::profile() const {
    std::map<std::string, double> kinds;
    for (const auto& [name, milliseconds] : graph->getProfilingResults())
        kinds[name] += milliseconds;
    std::vector<std::pair<std::string, double>> sorted(kinds.begin(), kinds.end());
    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    return sorted;
}

std::vector<std::pair<std::string, double>> PhotoEnhancerTraining::profileElements() const {
    std::vector<std::pair<std::string, double>> elements;
    for (const auto& [name, milliseconds] : graph->getProfilingResults())
        elements.emplace_back(name, milliseconds);
    return elements;
}

double PhotoEnhancerTraining::gpuSpan() const { return graph->getGpuSpanMs(); }

void PhotoEnhancerTraining::resetProfile() { graph->resetProfiling(); }

float PhotoEnhancerTraining::perceptualLoss() const { return vgg ? vgg->loss() : 0.0f; }

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

std::vector<float> PhotoEnhancerTraining::generatorInput() const {
    std::vector<float> values(input->getDataElementCount());
    input->getDataBuffer(0).memcopyTo(values);
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
    if (!config.forwardOnly && parameter.group != Discriminator)
        parameter.average->getDataBuffer(0).memcopyFrom(values);
}

} // namespace klartraum::neural
