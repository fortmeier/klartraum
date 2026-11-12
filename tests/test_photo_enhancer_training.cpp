// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - convBackwardMatchesPyTorch: the input, weight and bias gradients of 2D convolutions (3x3 with stride 1 and 2,
 *   1x1, 4x4 with stride 2) equal PyTorch autograd's for the same inputs and upstream gradient
 * - activationBackwardMatchesPyTorch: the ELU and leaky ReLU (0.2) input gradients, and the leaky ReLU itself, equal
 *   PyTorch's
 * - channelNormMatchesPyTorch: ChannelNorm's output and its input, scale and shift gradients equal PyTorch's
 *   (train_photo_enhancer.ChannelNorm)
 * - upsampleBackwardMatchesPyTorch: the input gradient of a bilinear upsampling by two equals PyTorch's
 *   (F.interpolate, align_corners=False)
 * - channelCopyConcatsAndSplits: copying two tensors' channels to their offsets concatenates them, copying them back
 *   out recovers them, and accumulating adds
 * - filmMatchesPyTorch: FiLM's output and its input, scale and shift gradients equal PyTorch's
 * - photometricLossMatchesPyTorch: the gradient of 0.5 L1 + 5 L1 of 8 x 8 block means equals PyTorch's, and the sum
 *   of the loss shares is the loss
 * - flowWarpLossMatchesPyTorch: the temporal consistency loss (the first render warped by a flow partly beyond the
 *   image, bilinear and clamped, against the second on a mask) and its gradient for the second render equal
 *   latent_renderer.flow_warp_loss()'s (grid_sample)
 * - linearMatchesPyTorch: a linear layer's output (with an offset) and its input (also accumulating), weight and
 *   bias gradients equal nn.Linear's
 * - embeddingPicksAndScattersRows: rows picked by index (one twice) come out in batch order; the table gradient adds
 *   the picked rows' gradients plus the L2 penalty's, and is zero for rows nobody picked
 * - clipGradientMatchesPyTorch: clipping two tensors to a global norm of 1 equals clip_grad_norm_
 * - emaCopiesThenAverages: during the warm-up the average copies the parameters, after it moves by the rate
 * - generatorStepMatchesPyTorch: one PhotoEnhancerTraining step of a 3-level generator with texture and FiLM
 *   appearance codes, from PyTorch's parameters on PyTorch's batch: the image and the photometric loss equal
 *   PyTorch's, and after the backward pass, clipping and Adam every parameter is within a tenth of its learning
 *   rate of PyTorch's (Adam's first step moves each by about the learning rate, so a wrong gradient shows)
 * - generatorStepWithVggMatchesPyTorch: the same step with the VGG perceptual loss (weight 1) also gives PyTorch's
 *   perceptual loss and updated parameters (skips without the VGG weights)
 * - ganStepMatchesPyTorch: one step with the 2-scale PatchGAN on 64 x 64 crops, the augmentations' numbers given:
 *   the discriminator's loss, the adversarial loss and the feature matching equal PyTorch's, and after the
 *   discriminator's and the generator's Adam steps all parameters of both are within a tenth of their learning rate
 *   of PyTorch's
 * - latentStepMatchesPyTorch: one step of the latent-conditioned renderer (4 geometry channels, texture, encoder,
 *   VAE with given noise, latent G-buffer, decoder; L1, block-mean L1, VGG and KL) from latent_renderer.py's
 *   parameters on its batch: image, losses and every parameter after clipping and Adam equal PyTorch's
 * - latentCheckStepMatchesPyTorch: the same step with the latent check (weight 0.5): every view rendered again with
 *   the other view's latents, encoded by the fixed encoder, L1 to those latents; the check loss, the image and every
 *   parameter after the step equal latent_renderer.latent_check()'s
 * - temporalStepMatchesPyTorch: the same step with the temporal consistency loss (weight 0.5): a second view of each
 *   sample rendered with the same latents, the first render warped onto it by a given flow and mask; the temporal
 *   loss, the image and every parameter after the step equal latent_renderer.flow_warp_loss()'s
 * - latentGanStepTrains: the latent-conditioned renderer with free bits and the 2-scale PatchGAN, from PyTorch's
 *   renderer and a fresh discriminator on its batch upsampled to 64 x 64, starts at the untrained discriminator's loss
 * (about 4 ln 2), keeps all losses finite over 30 steps on one batch, lowers the photometric loss and counts the KL at
 * least at its free bits
 * - forwardOnlyRendersTheSameImage: parameters saved from a training and loaded into a forward-only instance render
 *   PyTorch's image of the batch; with every row's code set to one code, the crops render as with that code
 * - maxPoolMatchesPyTorch: a 2 x 2 max-pool and its input gradient equal F.max_pool2d's
 * - vggLossMatchesPyTorch: the perceptual loss of an image against a target through VGG-16 up to relu4_3 and its
 *   gradient (added to a zeroed image gradient) equal train_photo_enhancer.Vgg's (skips without
 *   scripts/neural/export_vgg16.py's weights)
 * - objectMeanMatchesPyTorch: the mean over each object's pixels (an object missing in one sample) and the pixel
 *   counts, their backward pass (broadcast divided by the counts), the latent G-buffer (broadcast) and its backward
 *   pass (sum over the object's pixels) equal latent_renderer.object_mean()'s and splat()'s
 * - vaeMatchesPyTorch: the VAE's latents and KL divergence and the statistics' gradient equal PyTorch's
 * - vaeFreeBitsMatchesPyTorch: with 0.3 free bits, the KL term (each number counted as at least 0.3 nats) and the
 *   statistics' gradient (no KL gradient for numbers below) equal PyTorch's
 * - hashTextureBackwardMatchesPyTorch: for a batch of two images, the texture's features and its table gradient (the
 *   atomic additions into a zeroed gradient) equal PyTorch's (hash_texture.SurfaceTexture)
 *
 * The references come from scripts/neural/make_training_fixtures.py (build/TestingOutput/training_fixtures/); the
 * tests skip without them.
 **/

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/tensorelement.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/layers/training.hpp"
#include "klartraum/neural/hash_texture.hpp"
#include "klartraum/neural/photo_enhancer_training.hpp"
#include "klartraum/neural/vgg_loss.hpp"

using namespace klartraum;

namespace {

const std::string kFixtures = "build/TestingOutput/training_fixtures/";

// A fixture case: its tensors' shapes and values.
struct Fixture {
    std::map<std::string, std::vector<uint32_t>> shapes;

    explicit Fixture(const std::string& name)
        : name(name) {
        std::ifstream file(kFixtures + name + ".txt");
        for (std::string line; std::getline(file, line);) {
            std::istringstream words(line);
            std::string tensor;
            words >> tensor;
            std::vector<uint32_t> shape;
            for (uint32_t d; words >> d;)
                shape.push_back(d);
            shapes[tensor] = shape;
        }
    }

    bool exists() const { return !shapes.empty(); }

    std::vector<float> values(const std::string& tensor) const {
        std::ifstream file(kFixtures + name + "_" + tensor + ".bin", std::ios::binary | std::ios::ate);
        std::vector<float> data(size_t(file.tellg()) / sizeof(float));
        file.seekg(0);
        file.read(reinterpret_cast<char*>(data.data()), std::streamsize(data.size() * sizeof(float)));
        return data;
    }

    std::string name;
};

std::shared_ptr<TensorElement<float>> tensor(VulkanContext& vc, const std::vector<uint32_t>& shape) {
    return vc.create<TensorElement<float>>(shape);
}

std::vector<float> read(const std::shared_ptr<TensorElement<float>>& t) {
    std::vector<float> values(t->getDataElementCount());
    t->getDataBuffer(0).memcopyTo(values);
    return values;
}

// The largest difference relative to the reference's largest magnitude.
float relativeError(const std::vector<float>& actual, const std::vector<float>& expected) {
    EXPECT_EQ(actual.size(), expected.size());
    float worst = 0.0f, largest = 1e-12f;
    for (size_t i = 0; i < std::min(actual.size(), expected.size()); ++i) {
        worst = std::max(worst, std::abs(actual[i] - expected[i]));
        largest = std::max(largest, std::abs(expected[i]));
    }
    return worst / largest;
}

} // namespace

TEST(PhotoEnhancerTraining, convBackwardMatchesPyTorch) {
    const struct {
        const char* name;
        uint32_t kernel, stride, pad;
    } cases[] = {
        {"conv3x3", 3, 1, 1}, {"conv3x3_stride2", 3, 2, 1}, {"conv1x1", 1, 1, 0}, {"conv4x4_stride2", 4, 2, 1}};
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    for (const auto& c : cases) {
        const Fixture fixture(c.name);
        if (!fixture.exists())
            GTEST_SKIP() << "run scripts/neural/make_training_fixtures.py first";
        layers::ConvAttributes attributes;
        attributes.kernelShape = {c.kernel, c.kernel};
        attributes.strides = {c.stride, c.stride};
        attributes.pads = {c.pad, c.pad, c.pad, c.pad};
        const auto& xShape = fixture.shapes.at("x");
        const auto& wShape = fixture.shapes.at("w");
        const auto& yShape = fixture.shapes.at("y");
        auto x = tensor(vc, xShape), w = tensor(vc, wShape), dy = tensor(vc, yShape);
        auto dx = tensor(vc, xShape), dw = tensor(vc, wShape), db = tensor(vc, {wShape[0]});
        auto inputGradient = training::convBackwardInput(vc, attributes, xShape, wShape, yShape);
        inputGradient->setInput(dy, 0);
        inputGradient->setInput(w, 1);
        inputGradient->setInput(dx, 2);
        auto weightGradient = training::convBackwardWeights(vc, attributes, xShape, wShape, yShape);
        weightGradient->setInput(x, 0);
        weightGradient->setInput(dy, 1);
        weightGradient->setInput(dw, 2);
        weightGradient->setInput(db, 3);
        weightGradient->addDependency(inputGradient);
        ComputeGraph graph(vc, 1);
        graph.compileFrom(weightGradient);
        x->getDataBuffer(0).memcopyFrom(fixture.values("x"));
        w->getDataBuffer(0).memcopyFrom(fixture.values("w"));
        dy->getDataBuffer(0).memcopyFrom(fixture.values("dy"));
        graph.submitAndWait(vc.getGraphicsQueue(), 0);
        EXPECT_LT(relativeError(read(dx), fixture.values("dx")), 1e-5f) << c.name;
        EXPECT_LT(relativeError(read(dw), fixture.values("dw")), 1e-5f) << c.name;
        EXPECT_LT(relativeError(read(db), fixture.values("db")), 1e-5f) << c.name;
    }
}

namespace {

// Runs `element` on tensors of the given shapes (bound to its slots in order, the first `inputs` filled from the
// fixture's tensors of the same names) and returns the fixture comparisons of the rest.
void checkElement(VulkanContext& vc, const Fixture& fixture, ComputeGraphElementPtr element,
                  const std::vector<std::pair<std::string, std::vector<uint32_t>>>& slots, size_t inputs,
                  float tolerance = 1e-5f) {
    std::vector<std::shared_ptr<TensorElement<float>>> tensors;
    for (size_t k = 0; k < slots.size(); ++k) {
        tensors.push_back(tensor(vc, slots[k].second));
        element->setInput(tensors.back(), int(k));
    }
    ComputeGraph graph(vc, 1);
    graph.compileFrom(element);
    for (size_t k = 0; k < inputs; ++k)
        tensors[k]->getDataBuffer(0).memcopyFrom(fixture.values(slots[k].first));
    graph.submitAndWait(vc.getGraphicsQueue(), 0);
    for (size_t k = inputs; k < slots.size(); ++k)
        if (fixture.shapes.count(slots[k].first))
            EXPECT_LT(relativeError(read(tensors[k]), fixture.values(slots[k].first)), tolerance)
                << fixture.name << " " << slots[k].first;
}

} // namespace

TEST(PhotoEnhancerTraining, activationBackwardMatchesPyTorch) {
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    for (const auto& [name, activation] :
         {std::pair{"elu", training::Activation::Elu}, std::pair{"leaky_relu", training::Activation::LeakyRelu}}) {
        const Fixture fixture(name);
        if (!fixture.exists())
            GTEST_SKIP() << "run scripts/neural/make_training_fixtures.py first";
        const auto shape = fixture.shapes.at("x");
        checkElement(vc, fixture, training::activationBackward(vc, activation, shape),
                     {{"x", shape}, {"dy", shape}, {"dx", shape}}, 2);
        if (activation == training::Activation::LeakyRelu)
            checkElement(vc, fixture, training::leakyRelu(vc, shape), {{"x", shape}, {"y", shape}}, 1);
    }
}

TEST(PhotoEnhancerTraining, channelNormMatchesPyTorch) {
    const Fixture fixture("channel_norm");
    if (!fixture.exists())
        GTEST_SKIP() << "run scripts/neural/make_training_fixtures.py first";
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    const auto shape = fixture.shapes.at("x");
    const uint32_t c = shape[1];
    const std::vector<uint32_t> statsShape{shape[0], 2, shape[2], shape[3]};
    auto x = tensor(vc, shape), weight = tensor(vc, {c}), bias = tensor(vc, {c}), y = tensor(vc, shape);
    auto stats = tensor(vc, statsShape), dy = tensor(vc, shape), dx = tensor(vc, shape);
    auto dweight = tensor(vc, {c}), dbias = tensor(vc, {c});
    auto forward = training::channelNorm(vc, shape);
    forward->setInput(x, 0);
    forward->setInput(weight, 1);
    forward->setInput(bias, 2);
    forward->setInput(y, 3);
    forward->setInput(stats, 4);
    auto input = training::channelNormBackwardInput(vc, shape);
    input->setInput(x, 0);
    input->setInput(weight, 1);
    input->setInput(stats, 2);
    input->setInput(dy, 3);
    input->setInput(dx, 4);
    input->addDependency(forward);
    auto parameters = training::channelNormBackwardParameters(vc, shape);
    parameters->setInput(x, 0);
    parameters->setInput(stats, 1);
    parameters->setInput(dy, 2);
    parameters->setInput(dweight, 3);
    parameters->setInput(dbias, 4);
    parameters->addDependency(input);
    ComputeGraph graph(vc, 1);
    graph.compileFrom(parameters);
    x->getDataBuffer(0).memcopyFrom(fixture.values("x"));
    weight->getDataBuffer(0).memcopyFrom(fixture.values("weight"));
    bias->getDataBuffer(0).memcopyFrom(fixture.values("bias"));
    dy->getDataBuffer(0).memcopyFrom(fixture.values("dy"));
    graph.submitAndWait(vc.getGraphicsQueue(), 0);
    EXPECT_LT(relativeError(read(y), fixture.values("y")), 1e-5f);
    EXPECT_LT(relativeError(read(dx), fixture.values("dx")), 1e-5f);
    EXPECT_LT(relativeError(read(dweight), fixture.values("dweight")), 1e-5f);
    EXPECT_LT(relativeError(read(dbias), fixture.values("dbias")), 1e-5f);
}

TEST(PhotoEnhancerTraining, upsampleBackwardMatchesPyTorch) {
    const Fixture fixture("upsample2x");
    if (!fixture.exists())
        GTEST_SKIP() << "run scripts/neural/make_training_fixtures.py first";
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    const auto shape = fixture.shapes.at("x");
    checkElement(vc, fixture, training::upsample2xBackward(vc, shape), {{"dy", fixture.shapes.at("y")}, {"dx", shape}},
                 1);
}

TEST(PhotoEnhancerTraining, channelCopyConcatsAndSplits) {
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    const uint32_t n = 2, pixels = 6;
    auto a = tensor(vc, {n, 2, 2, 3}), b = tensor(vc, {n, 3, 2, 3}), joined = tensor(vc, {n, 5, 2, 3});
    auto back = tensor(vc, {n, 3, 2, 3});
    auto copyA = training::channelCopy(vc, n, pixels, {2, 0}, {5, 0}, 2);
    copyA->setInput(a, 0);
    copyA->setInput(joined, 1);
    auto copyB = training::channelCopy(vc, n, pixels, {3, 0}, {5, 2}, 3);
    copyB->setInput(b, 0);
    copyB->setInput(joined, 1);
    copyB->addDependency(copyA);
    // b's channels back out, then added once more.
    auto split = training::channelCopy(vc, n, pixels, {5, 2}, {3, 0}, 3);
    split->setInput(joined, 0);
    split->setInput(back, 1);
    split->addDependency(copyB);
    auto add = training::channelCopy(vc, n, pixels, {5, 2}, {3, 0}, 3, true);
    add->setInput(joined, 0);
    add->setInput(back, 1);
    add->addDependency(split);
    ComputeGraph graph(vc, 1);
    graph.compileFrom(add);
    std::vector<float> va(n * 2 * pixels), vb(n * 3 * pixels);
    for (size_t i = 0; i < va.size(); ++i)
        va[i] = float(i);
    for (size_t i = 0; i < vb.size(); ++i)
        vb[i] = 100.0f + float(i);
    a->getDataBuffer(0).memcopyFrom(va);
    b->getDataBuffer(0).memcopyFrom(vb);
    graph.submitAndWait(vc.getGraphicsQueue(), 0);
    const auto j = read(joined), r = read(back);
    for (uint32_t s = 0; s < n; ++s)
        for (uint32_t p = 0; p < pixels; ++p) {
            for (uint32_t c = 0; c < 2; ++c)
                EXPECT_EQ(j[(s * 5 + c) * pixels + p], va[(s * 2 + c) * pixels + p]);
            for (uint32_t c = 0; c < 3; ++c) {
                EXPECT_EQ(j[(s * 5 + 2 + c) * pixels + p], vb[(s * 3 + c) * pixels + p]);
                EXPECT_EQ(r[(s * 3 + c) * pixels + p], 2.0f * vb[(s * 3 + c) * pixels + p]);
            }
        }
}

TEST(PhotoEnhancerTraining, filmMatchesPyTorch) {
    const Fixture fixture("film");
    if (!fixture.exists())
        GTEST_SKIP() << "run scripts/neural/make_training_fixtures.py first";
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    const auto shape = fixture.shapes.at("x");
    const std::vector<uint32_t> planes{shape[0], shape[1]};
    checkElement(vc, fixture, training::film(vc, shape),
                 {{"x", shape}, {"scale", planes}, {"shift", planes}, {"y", shape}}, 3);
    checkElement(
        vc, fixture, training::filmBackward(vc, shape),
        {{"x", shape}, {"scale", planes}, {"dy", shape}, {"dx", shape}, {"dscale", planes}, {"dshift", planes}}, 3);
}

TEST(PhotoEnhancerTraining, hashTextureBackwardMatchesPyTorch) {
    const Fixture fixture("hash_texture");
    if (!fixture.exists())
        GTEST_SKIP() << "run scripts/neural/make_training_fixtures.py first";
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    auto shape = neural::HashTextureShape::fromFile(kFixtures + "hash_texture_shape.txt");
    shape.batch = 2;
    auto coordinates = tensor(vc, fixture.shapes.at("coordinates")), classes = tensor(vc, fixture.shapes.at("classes"));
    auto table = tensor(vc, fixture.shapes.at("table")), features = tensor(vc, fixture.shapes.at("features"));
    auto dfeatures = tensor(vc, fixture.shapes.at("dfeatures")), dtable = tensor(vc, fixture.shapes.at("table"));
    auto forward = neural::hashTexture(vc, shape);
    forward->setInput(coordinates, 0);
    forward->setInput(classes, 1);
    forward->setInput(table, 2);
    forward->setInput(features, 3);
    auto zero = training::fill(vc, uint32_t(dtable->getDataElementCount()));
    zero->setInput(dtable, 0);
    zero->addDependency(forward);
    auto backward = neural::hashTextureBackward(vc, shape);
    backward->setInput(coordinates, 0);
    backward->setInput(classes, 1);
    backward->setInput(dfeatures, 2);
    backward->setInput(dtable, 3);
    backward->addDependency(zero);
    ComputeGraph graph(vc, 1);
    graph.compileFrom(backward);
    coordinates->getDataBuffer(0).memcopyFrom(fixture.values("coordinates"));
    classes->getDataBuffer(0).memcopyFrom(fixture.values("classes"));
    table->getDataBuffer(0).memcopyFrom(fixture.values("table"));
    dfeatures->getDataBuffer(0).memcopyFrom(fixture.values("dfeatures"));
    // Twice: the gradient is zeroed every submission, so it must not accumulate.
    graph.submitAndWait(vc.getGraphicsQueue(), 0);
    graph.submitAndWait(vc.getGraphicsQueue(), 0);
    EXPECT_LT(relativeError(read(features), fixture.values("features")), 1e-5f);
    EXPECT_LT(relativeError(read(dtable), fixture.values("dtable")), 1e-5f);
}

TEST(PhotoEnhancerTraining, photometricLossMatchesPyTorch) {
    const Fixture fixture("photometric");
    if (!fixture.exists())
        GTEST_SKIP() << "run scripts/neural/make_training_fixtures.py first";
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    const auto shape = fixture.shapes.at("image");
    auto image = tensor(vc, shape), target = tensor(vc, shape), gradient = tensor(vc, shape);
    auto shares = tensor(vc, shape), total = tensor(vc, {1});
    auto loss = training::photometricLoss(vc, shape, 0.5f, 5.0f);
    loss->setInput(image, 0);
    loss->setInput(target, 1);
    loss->setInput(gradient, 2);
    loss->setInput(shares, 3);
    auto add = training::sum(vc, uint32_t(shares->getDataElementCount()));
    add->setInput(shares, 0);
    add->setInput(total, 1);
    add->addDependency(loss);
    ComputeGraph graph(vc, 1);
    graph.compileFrom(add);
    image->getDataBuffer(0).memcopyFrom(fixture.values("image"));
    target->getDataBuffer(0).memcopyFrom(fixture.values("target"));
    graph.submitAndWait(vc.getGraphicsQueue(), 0);
    EXPECT_LT(relativeError(read(gradient), fixture.values("dimage")), 1e-5f);
    EXPECT_NEAR(read(total)[0], fixture.values("loss")[0], 1e-5f);
}

TEST(PhotoEnhancerTraining, flowWarpLossMatchesPyTorch) {
    const Fixture fixture("flow_warp");
    if (!fixture.exists())
        GTEST_SKIP() << "run scripts/neural/make_training_fixtures.py first";
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    const auto shape = fixture.shapes.at("second");
    auto first = tensor(vc, shape), second = tensor(vc, shape), gradient = tensor(vc, shape);
    auto flow = tensor(vc, fixture.shapes.at("flow")), valid = tensor(vc, fixture.shapes.at("valid"));
    auto shares = tensor(vc, shape), total = tensor(vc, {1});
    auto loss = training::flowWarpLoss(vc, shape, 0.7f);
    loss->setInput(first, 0);
    loss->setInput(second, 1);
    loss->setInput(flow, 2);
    loss->setInput(valid, 3);
    loss->setInput(gradient, 4);
    loss->setInput(shares, 5);
    auto add = training::sum(vc, uint32_t(shares->getDataElementCount()));
    add->setInput(shares, 0);
    add->setInput(total, 1);
    add->addDependency(loss);
    ComputeGraph graph(vc, 1);
    graph.compileFrom(add);
    for (const auto& [element, name] :
         {std::pair{first, "first"}, std::pair{second, "second"}, std::pair{flow, "flow"}, std::pair{valid, "valid"}})
        element->getDataBuffer(0).memcopyFrom(fixture.values(name));
    graph.submitAndWait(vc.getGraphicsQueue(), 0);
    EXPECT_LT(relativeError(read(gradient), fixture.values("dsecond")), 1e-5f);
    EXPECT_NEAR(read(total)[0], fixture.values("loss")[0], 1e-6f);
}

TEST(PhotoEnhancerTraining, linearMatchesPyTorch) {
    const Fixture fixture("linear");
    if (!fixture.exists())
        GTEST_SKIP() << "run scripts/neural/make_training_fixtures.py first";
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    const training::LinearShape shape{3, 5, 7};
    checkElement(vc, fixture, training::linear(vc, shape), {{"x", {3, 5}}, {"w", {7, 5}}, {"b", {7}}, {"y", {3, 7}}},
                 3);
    checkElement(vc, fixture, training::linearBackwardInput(vc, shape), {{"dy", {3, 7}}, {"w", {7, 5}}, {"dx", {3, 5}}},
                 2);
    checkElement(vc, fixture, training::linearBackwardWeights(vc, shape),
                 {{"x", {3, 5}}, {"dy", {3, 7}}, {"dw", {7, 5}}, {"db", {7}}}, 2);
    // The offset adds to every output, and accumulate adds to the input gradient.
    auto x = tensor(vc, {3, 5}), w = tensor(vc, {7, 5}), b = tensor(vc, {7}), y = tensor(vc, {3, 7});
    auto dy = tensor(vc, {3, 7}), dx = tensor(vc, {3, 5});
    auto forward = training::linear(vc, shape, 1.0f);
    forward->setInput(x, 0);
    forward->setInput(w, 1);
    forward->setInput(b, 2);
    forward->setInput(y, 3);
    auto first = training::linearBackwardInput(vc, shape);
    first->setInput(dy, 0);
    first->setInput(w, 1);
    first->setInput(dx, 2);
    first->addDependency(forward);
    auto second = training::linearBackwardInput(vc, shape, true);
    second->setInput(dy, 0);
    second->setInput(w, 1);
    second->setInput(dx, 2);
    second->addDependency(first);
    ComputeGraph graph(vc, 1);
    graph.compileFrom(second);
    x->getDataBuffer(0).memcopyFrom(fixture.values("x"));
    w->getDataBuffer(0).memcopyFrom(fixture.values("w"));
    b->getDataBuffer(0).memcopyFrom(fixture.values("b"));
    dy->getDataBuffer(0).memcopyFrom(fixture.values("dy"));
    graph.submitAndWait(vc.getGraphicsQueue(), 0);
    auto expectedY = fixture.values("y"), expectedDx = fixture.values("dx");
    for (auto& v : expectedY)
        v += 1.0f;
    for (auto& v : expectedDx)
        v *= 2.0f;
    EXPECT_LT(relativeError(read(y), expectedY), 1e-5f);
    EXPECT_LT(relativeError(read(dx), expectedDx), 1e-5f);
}

TEST(PhotoEnhancerTraining, embeddingPicksAndScattersRows) {
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    const uint32_t batch = 3, rows = 4, size = 2;
    const float decay = 0.1f;
    auto table = tensor(vc, {rows, size}), indices = tensor(vc, {batch}), picked = tensor(vc, {batch, size});
    auto pickedGradient = tensor(vc, {batch, size}), tableGradient = tensor(vc, {rows, size});
    auto pick = training::embedding(vc, batch, rows, size);
    pick->setInput(table, 0);
    pick->setInput(indices, 1);
    pick->setInput(picked, 2);
    auto scatter = training::embeddingBackward(vc, batch, rows, size, decay);
    scatter->setInput(table, 0);
    scatter->setInput(indices, 1);
    scatter->setInput(pickedGradient, 2);
    scatter->setInput(tableGradient, 3);
    scatter->addDependency(pick);
    ComputeGraph graph(vc, 1);
    graph.compileFrom(scatter);
    const std::vector<float> values{1, 2, 3, 4, 5, 6, 7, 8}, rowsPicked{2, 0, 2}, gradients{0.5f, 1, 2, 3, 4, 5};
    table->getDataBuffer(0).memcopyFrom(values);
    indices->getDataBuffer(0).memcopyFrom(rowsPicked);
    pickedGradient->getDataBuffer(0).memcopyFrom(gradients);
    graph.submitAndWait(vc.getGraphicsQueue(), 0);
    EXPECT_EQ(read(picked), (std::vector<float>{5, 6, 1, 2, 5, 6}));
    const auto g = read(tableGradient);
    const float p = 2.0f * decay / float(batch);
    EXPECT_NEAR(g[0], 2.0f + p * 1.0f, 1e-6f);
    EXPECT_NEAR(g[1], 3.0f + p * 2.0f, 1e-6f);
    EXPECT_EQ(g[2], 0.0f);
    EXPECT_EQ(g[3], 0.0f);
    EXPECT_NEAR(g[4], 0.5f + 4.0f + 2.0f * p * 5.0f, 1e-6f);
    EXPECT_NEAR(g[5], 1.0f + 5.0f + 2.0f * p * 6.0f, 1e-6f);
    EXPECT_EQ(g[6], 0.0f);
}

TEST(PhotoEnhancerTraining, clipGradientMatchesPyTorch) {
    const Fixture fixture("clip");
    if (!fixture.exists())
        GTEST_SKIP() << "run scripts/neural/make_training_fixtures.py first";
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    auto a = tensor(vc, {300}), b = tensor(vc, {70}), total = tensor(vc, {1});
    auto zero = training::fill(vc, 1);
    zero->setInput(total, 0);
    auto normA = training::squaredNorm(vc, 300);
    normA->setInput(a, 0);
    normA->setInput(total, 1);
    normA->addDependency(zero);
    auto normB = training::squaredNorm(vc, 70);
    normB->setInput(b, 0);
    normB->setInput(total, 1);
    normB->addDependency(normA);
    auto clipA = training::clipGradient(vc, 300, 1.0f);
    clipA->setInput(a, 0);
    clipA->setInput(total, 1);
    clipA->addDependency(normB);
    auto clipB = training::clipGradient(vc, 70, 1.0f);
    clipB->setInput(b, 0);
    clipB->setInput(total, 1);
    clipB->addDependency(clipA);
    ComputeGraph graph(vc, 1);
    graph.compileFrom(clipB);
    a->getDataBuffer(0).memcopyFrom(fixture.values("ga"));
    b->getDataBuffer(0).memcopyFrom(fixture.values("gb"));
    graph.submitAndWait(vc.getGraphicsQueue(), 0);
    EXPECT_LT(relativeError(read(a), fixture.values("clipped_a")), 1e-5f);
    EXPECT_LT(relativeError(read(b), fixture.values("clipped_b")), 1e-5f);
}

TEST(PhotoEnhancerTraining, emaCopiesThenAverages) {
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    auto average = vc.create<TensorElement<float>>(std::vector<uint32_t>{2});
    auto parameters = vc.create<TensorElement<float>>(std::vector<uint32_t>{2});
    auto counter = vc.create<TensorElement<uint32_t>>(std::vector<uint32_t>{1});
    auto step = training::stepCounter(vc);
    step->setInput(counter, 0);
    auto update = training::ema(vc, 2, 0.25f, 1);
    update->setInput(average, 0);
    update->setInput(parameters, 1);
    update->setInput(counter, 2);
    update->addDependency(step);
    ComputeGraph graph(vc, 1);
    graph.compileFrom(update);
    average->getDataBuffer(0).memcopyFrom(std::vector<float>{0.0f, 0.0f});
    counter->getDataBuffer(0).memcopyFrom(std::vector<uint32_t>{0u});
    parameters->getDataBuffer(0).memcopyFrom(std::vector<float>{4.0f, 8.0f});
    graph.submitAndWait(vc.getGraphicsQueue(), 0); // step 1: copy
    EXPECT_EQ(read(average), (std::vector<float>{4.0f, 8.0f}));
    parameters->getDataBuffer(0).memcopyFrom(std::vector<float>{8.0f, 0.0f});
    graph.submitAndWait(vc.getGraphicsQueue(), 0); // step 2: a quarter of the way
    EXPECT_EQ(read(average), (std::vector<float>{5.0f, 6.0f}));
}

namespace {

// One PhotoEnhancerTraining step against the fixture `name` of make_training_fixtures.generator_step_case().
void checkGeneratorStep(const std::string& name, float perceptual) {
    const Fixture fixture(name);
    if (!fixture.exists())
        GTEST_SKIP() << "run scripts/neural/make_training_fixtures.py first";
    if (perceptual > 0.0f && !std::filesystem::exists("build/models/vgg16/layers.txt"))
        GTEST_SKIP() << "run scripts/neural/export_vgg16.py first";
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    neural::PhotoEnhancerTrainingConfig config;
    config.batch = 2;
    config.size = 32;
    config.widths = {32, 64, 128};
    config.texture = neural::HashTextureShape::fromFile(kFixtures + name + "_texture.txt");
    config.appearances = 5;
    config.appearanceSize = 8;
    config.perceptual = perceptual;
    neural::PhotoEnhancerTraining trainer(vc, config);
    std::vector<std::string> names;
    {
        std::ifstream file(kFixtures + name + "_parameters.txt");
        for (std::string parameter; file >> parameter;)
            names.push_back(parameter);
    }
    EXPECT_EQ(trainer.parameterNames(), names);
    for (const auto& parameter : names)
        trainer.setParameter(parameter, fixture.values("before." + parameter));
    trainer.setBatch(fixture.values("channels"), fixture.values("coordinates"), fixture.values("classes"),
                     fixture.values("target"), fixture.values("rows"));
    trainer.step();
    EXPECT_LT(relativeError(trainer.image(), fixture.values("image")), 1e-5f);
    // PyTorch's loss includes the codes' penalty 1e-2 x mean |code|^2 of rows 3 and 1.
    const auto codes = fixture.values("before.codes.weight");
    float penalty = 0.0f;
    for (int row : {3, 1})
        for (int k = 0; k < 8; ++k)
            penalty += codes[row * 8 + k] * codes[row * 8 + k];
    EXPECT_NEAR(trainer.loss() + trainer.perceptualLoss(), fixture.values("loss")[0] - 1e-2f * penalty / 2.0f, 1e-4f);
    for (const auto& parameter : names) {
        const auto before = fixture.values("before." + parameter), after = fixture.values("after." + parameter);
        std::vector<float> actual(before.size());
        trainer.parameter(parameter)->getDataBuffer(0).memcopyTo(actual);
        const float rate = parameter == "texture.table" ? 1e-2f : parameter == "codes.weight" ? 5e-3f : 1e-4f;
        float worst = 0.0f;
        for (size_t i = 0; i < actual.size(); ++i)
            worst = std::max(worst, std::abs(actual[i] - after[i]));
        EXPECT_LT(worst, 0.1f * rate) << parameter;
    }
}

} // namespace

TEST(PhotoEnhancerTraining, generatorStepMatchesPyTorch) { checkGeneratorStep("generator_step", 0.0f); }

TEST(PhotoEnhancerTraining, generatorStepWithVggMatchesPyTorch) { checkGeneratorStep("generator_step_vgg", 1.0f); }

TEST(PhotoEnhancerTraining, forwardOnlyRendersTheSameImage) {
    const Fixture fixture("generator_step");
    if (!fixture.exists())
        GTEST_SKIP() << "run scripts/neural/make_training_fixtures.py first";
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    neural::PhotoEnhancerTrainingConfig config;
    config.batch = 2;
    config.size = 32;
    config.texture = neural::HashTextureShape::fromFile(kFixtures + "generator_step_texture.txt");
    config.appearances = 5;
    config.appearanceSize = 8;
    const std::string directory = "build/TestingOutput/training_fixtures/saved_generator";
    {
        neural::PhotoEnhancerTraining trainer(vc, config);
        for (const auto& name : trainer.parameterNames())
            trainer.setParameter(name, fixture.values("before." + name));
        trainer.save(directory, false);
    }
    config.forwardOnly = true;
    neural::PhotoEnhancerTraining renderer(vc, config);
    renderer.load(directory);
    renderer.setBatch(fixture.values("channels"), fixture.values("coordinates"), fixture.values("classes"),
                      fixture.values("target"), fixture.values("rows"));
    renderer.step();
    EXPECT_LT(relativeError(renderer.image(), fixture.values("image")), 1e-5f);
    // Row 3's code for every crop: the first crop (row 3) stays, the second (row 1) changes.
    const auto codes = fixture.values("before.codes.weight");
    renderer.setAppearance(std::vector<float>(codes.begin() + 3 * 8, codes.begin() + 4 * 8));
    renderer.step();
    const auto image = renderer.image(), expected = fixture.values("image");
    const size_t half = image.size() / 2;
    EXPECT_LT(relativeError({image.begin(), image.begin() + half}, {expected.begin(), expected.begin() + half}), 1e-5f);
    EXPECT_GT(relativeError({image.begin() + half, image.end()}, {expected.begin() + half, expected.end()}), 1e-3f);
}

TEST(PhotoEnhancerTraining, maxPoolMatchesPyTorch) {
    const Fixture fixture("max_pool");
    if (!fixture.exists())
        GTEST_SKIP() << "run scripts/neural/make_training_fixtures.py first";
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    const auto shape = fixture.shapes.at("x"), pooled = fixture.shapes.at("y");
    auto x = tensor(vc, shape), y = tensor(vc, pooled), taken = tensor(vc, pooled), dy = tensor(vc, pooled);
    auto dx = tensor(vc, shape);
    auto forward = training::maxPool(vc, shape);
    forward->setInput(x, 0);
    forward->setInput(y, 1);
    forward->setInput(taken, 2);
    auto backward = training::maxPoolBackward(vc, shape);
    backward->setInput(dy, 0);
    backward->setInput(taken, 1);
    backward->setInput(dx, 2);
    backward->addDependency(forward);
    ComputeGraph graph(vc, 1);
    graph.compileFrom(backward);
    x->getDataBuffer(0).memcopyFrom(fixture.values("x"));
    dy->getDataBuffer(0).memcopyFrom(fixture.values("dy"));
    graph.submitAndWait(vc.getGraphicsQueue(), 0);
    EXPECT_EQ(read(y), fixture.values("y"));
    EXPECT_EQ(read(dx), fixture.values("dx"));
}

TEST(PhotoEnhancerTraining, vggLossMatchesPyTorch) {
    const Fixture fixture("vgg");
    const std::string weights = "build/models/vgg16";
    if (!fixture.exists() || !std::filesystem::exists(weights + "/layers.txt"))
        GTEST_SKIP() << "run scripts/neural/export_vgg16.py and make_training_fixtures.py first";
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    const auto shape = fixture.shapes.at("image");
    auto images = tensor(vc, shape), targets = tensor(vc, shape), gradient = tensor(vc, shape);
    auto zero = training::fill(vc, uint32_t(gradient->getDataElementCount()));
    zero->setInput(gradient, 0);
    neural::VggLoss loss(vc, weights, shape[0], shape[2], shape[3], 1.0f, images, targets, gradient);
    loss.elements().front()->addDependency(zero);
    ComputeGraph graph(vc, 1);
    graph.compileFrom(loss.elements().back());
    loss.upload();
    images->getDataBuffer(0).memcopyFrom(fixture.values("image"));
    targets->getDataBuffer(0).memcopyFrom(fixture.values("target"));
    graph.submitAndWait(vc.getGraphicsQueue(), 0);
    EXPECT_NEAR(loss.loss(), fixture.values("loss")[0], 1e-4f * fixture.values("loss")[0]);
    EXPECT_LT(relativeError(read(gradient), fixture.values("dimage")), 1e-3f);
}

TEST(PhotoEnhancerTraining, ganStepMatchesPyTorch) {
    const Fixture fixture("gan_step");
    if (!fixture.exists())
        GTEST_SKIP() << "run scripts/neural/make_training_fixtures.py first";
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    neural::PhotoEnhancerTrainingConfig config;
    config.batch = 2;
    config.size = 64;
    config.widths = {32, 64, 128};
    config.texture = neural::HashTextureShape::fromFile(kFixtures + "gan_step_texture.txt");
    config.appearances = 5;
    config.appearanceSize = 8;
    config.adversarial = 0.3f;
    config.featureMatching = 2.0f;
    neural::PhotoEnhancerTraining trainer(vc, config);
    std::vector<std::string> names;
    {
        std::ifstream file(kFixtures + "gan_step_parameters.txt");
        for (std::string parameter; file >> parameter;)
            names.push_back(parameter);
    }
    EXPECT_EQ(trainer.parameterNames(), names);
    for (const auto& parameter : names)
        trainer.setParameter(parameter, fixture.values("before." + parameter));
    trainer.setBatch(fixture.values("channels"), fixture.values("coordinates"), fixture.values("classes"),
                     fixture.values("target"), fixture.values("rows"));
    trainer.setAugmentation(fixture.values("augmentation"));
    trainer.step();
    EXPECT_LT(relativeError(trainer.image(), fixture.values("image")), 1e-5f);
    EXPECT_NEAR(trainer.discriminatorLoss(), fixture.values("loss_d")[0], 1e-4f);
    EXPECT_NEAR(trainer.adversarialLoss(), fixture.values("loss_adversarial")[0], 1e-4f);
    EXPECT_NEAR(trainer.featureMatchingLoss(), fixture.values("loss_feature_matching")[0], 1e-4f);
    for (const auto& parameter : names) {
        const auto after = fixture.values("after." + parameter);
        std::vector<float> actual(after.size());
        trainer.parameter(parameter)->getDataBuffer(0).memcopyTo(actual);
        const float rate = parameter == "texture.table"                ? 1e-2f
                           : parameter == "codes.weight"               ? 5e-3f
                           : parameter.rfind("discriminator.", 0) == 0 ? 2e-4f
                                                                       : 1e-4f;
        float worst = 0.0f;
        for (size_t i = 0; i < actual.size(); ++i)
            worst = std::max(worst, std::abs(actual[i] - after[i]));
        EXPECT_LT(worst, 0.1f * rate) << parameter;
    }
}

TEST(PhotoEnhancerTraining, objectMeanMatchesPyTorch) {
    const Fixture fixture("object_mean");
    if (!fixture.exists())
        GTEST_SKIP() << "run scripts/neural/make_training_fixtures.py first";
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    const auto shape = fixture.shapes.at("values");
    const training::ObjectSampling sampling{shape[3], shape[2], 1, 0};
    checkElement(vc, fixture, training::objectMean(vc, shape, 2, sampling),
                 {{"values", shape}, {"objects", {2, 5, 7}}, {"means", {2, 2, 6}}, {"counts", {2, 2}}}, 2);
    // The mean's backward pass: the gradient divided by the counts, at the object's pixels.
    {
        auto vectors = tensor(vc, {2, 2, 6}), objects = tensor(vc, {2, 5, 7}), counts = tensor(vc, {2, 2});
        auto output = tensor(vc, shape);
        auto broadcast = training::objectBroadcast(vc, shape, 2, sampling, 6, 0, true);
        broadcast->setInput(vectors, 0);
        broadcast->setInput(objects, 1);
        broadcast->setInput(counts, 2);
        broadcast->setInput(output, 3);
        ComputeGraph graph(vc, 1);
        graph.compileFrom(broadcast);
        vectors->getDataBuffer(0).memcopyFrom(fixture.values("dmeans"));
        objects->getDataBuffer(0).memcopyFrom(fixture.values("objects"));
        counts->getDataBuffer(0).memcopyFrom(fixture.values("counts"));
        graph.submitAndWait(vc.getGraphicsQueue(), 0);
        EXPECT_LT(relativeError(read(output), fixture.values("dvalues")), 1e-5f);
    }
    // The latent G-buffer (3 numbers per object into channels 1..3 of 5) and its backward pass.
    {
        const std::vector<uint32_t> grid{2, 3, 5, 7};
        auto latents = tensor(vc, {2, 2, 3}), objects = tensor(vc, {2, 5, 7}), counts = tensor(vc, {2, 2});
        auto gbuffer = tensor(vc, {2, 5, 5, 7}), gradient = tensor(vc, grid), latentGradient = tensor(vc, {2, 2, 3});
        auto sums = tensor(vc, {2, 2});
        auto broadcast = training::objectBroadcast(vc, grid, 2, sampling, 5, 1);
        broadcast->setInput(latents, 0);
        broadcast->setInput(objects, 1);
        broadcast->setInput(counts, 2);
        broadcast->setInput(gbuffer, 3);
        auto back = training::objectMean(vc, grid, 2, sampling, false);
        back->setInput(gradient, 0);
        back->setInput(objects, 1);
        back->setInput(latentGradient, 2);
        back->setInput(sums, 3);
        back->addDependency(broadcast);
        ComputeGraph graph(vc, 1);
        graph.compileFrom(back);
        latents->getDataBuffer(0).memcopyFrom(fixture.values("latents"));
        objects->getDataBuffer(0).memcopyFrom(fixture.values("objects"));
        counts->getDataBuffer(0).memcopyFrom(fixture.values("counts"));
        gbuffer->getDataBuffer(0).memcopyFrom(std::vector<float>(2 * 5 * 35, -1.0f));
        gradient->getDataBuffer(0).memcopyFrom(fixture.values("dsplatted"));
        graph.submitAndWait(vc.getGraphicsQueue(), 0);
        const auto written = read(gbuffer), expected = fixture.values("splatted");
        for (uint32_t n = 0; n < 2; ++n)
            for (uint32_t c = 0; c < 5; ++c)
                for (uint32_t p = 0; p < 35; ++p) {
                    const float value = written[(n * 5 + c) * 35 + p];
                    if (c == 0 || c == 4)
                        EXPECT_EQ(value, -1.0f);
                    else
                        EXPECT_NEAR(value, expected[(n * 3 + c - 1) * 35 + p], 1e-6f);
                }
        EXPECT_LT(relativeError(read(latentGradient), fixture.values("dlatents")), 1e-5f);
    }
}

namespace {

// Runs vae() and vaeBackward() on the "vae" fixture's statistics with @p freeBits and compares the KL divergence and
// the statistics' gradient with @p expected's (the latents with "vae"'s).
void expectVaeMatches(const Fixture& fixture, const Fixture& expected, float freeBits) {
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    // The statistics as the encoder pools them: per object the means, then the log-variances.
    const auto mu = fixture.values("mu"), lv = fixture.values("log_variance");
    std::vector<float> statistics;
    for (size_t object = 0; object < 4; ++object) {
        statistics.insert(statistics.end(), mu.begin() + object * 3, mu.begin() + object * 3 + 3);
        statistics.insert(statistics.end(), lv.begin() + object * 3, lv.begin() + object * 3 + 3);
    }
    auto stats = tensor(vc, {2, 2, 6}), epsilon = tensor(vc, {2, 2, 3}), counts = tensor(vc, {2, 2});
    auto latents = tensor(vc, {2, 2, 3}), shares = tensor(vc, {2, 2}), dlatents = tensor(vc, {2, 2, 3});
    auto dstats = tensor(vc, {2, 2, 6}), total = tensor(vc, {1});
    auto forward = training::vae(vc, 2, 2, 3, 0.3f, freeBits);
    forward->setInput(stats, 0);
    forward->setInput(epsilon, 1);
    forward->setInput(counts, 2);
    forward->setInput(latents, 3);
    forward->setInput(shares, 4);
    auto add = training::sum(vc, 4);
    add->setInput(shares, 0);
    add->setInput(total, 1);
    add->addDependency(forward);
    auto backward = training::vaeBackward(vc, 2, 2, 3, 0.3f, freeBits);
    backward->setInput(stats, 0);
    backward->setInput(epsilon, 1);
    backward->setInput(counts, 2);
    backward->setInput(dlatents, 3);
    backward->setInput(dstats, 4);
    backward->addDependency(add);
    ComputeGraph graph(vc, 1);
    graph.compileFrom(backward);
    stats->getDataBuffer(0).memcopyFrom(statistics);
    epsilon->getDataBuffer(0).memcopyFrom(fixture.values("epsilon"));
    counts->getDataBuffer(0).memcopyFrom(fixture.values("counts"));
    dlatents->getDataBuffer(0).memcopyFrom(fixture.values("dsample"));
    graph.submitAndWait(vc.getGraphicsQueue(), 0);
    EXPECT_LT(relativeError(read(latents), fixture.values("sample")), 1e-5f);
    EXPECT_NEAR(read(total)[0], 0.3f * expected.values("kl")[0], 1e-5f);
    const auto gradient = read(dstats), dmu = expected.values("dmu"), dlv = expected.values("dlog_variance");
    for (size_t object = 0; object < 4; ++object)
        for (size_t d = 0; d < 3; ++d) {
            EXPECT_NEAR(gradient[object * 6 + d], dmu[object * 3 + d], 1e-5f);
            EXPECT_NEAR(gradient[object * 6 + 3 + d], dlv[object * 3 + d], 1e-5f);
        }
}

} // namespace

TEST(PhotoEnhancerTraining, vaeMatchesPyTorch) {
    const Fixture fixture("vae");
    if (!fixture.exists())
        GTEST_SKIP() << "run scripts/neural/make_training_fixtures.py first";
    expectVaeMatches(fixture, fixture, 0.0f);
}

TEST(PhotoEnhancerTraining, vaeFreeBitsMatchesPyTorch) {
    const Fixture fixture("vae"), expected("vae_free_bits");
    if (!fixture.exists() || !expected.exists())
        GTEST_SKIP() << "run scripts/neural/make_training_fixtures.py first";
    expectVaeMatches(fixture, expected, 0.3f);
}

namespace {

// One step of the latent-conditioned renderer from latent_renderer.py's parameters on its batch (fixture @p name),
// with the latent check of weight @p check (and the fixture's permutation) and the temporal consistency loss of
// weight @p temporal (and the fixture's second views and flow) if positive: image, losses and every parameter after
// clipping and Adam equal PyTorch's.
void expectLatentStepMatches(const Fixture& fixture, const std::string& name, float check, float temporal = 0.0f) {
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    neural::PhotoEnhancerTrainingConfig config;
    config.batch = 2;
    config.size = 32;
    config.widths = {32, 64, 128};
    config.gbufferChannels = 4;
    config.texture = neural::HashTextureShape::fromFile(kFixtures + name + "_texture.txt");
    config.latentSize = 4;
    config.latentObjects = 1;
    config.klWeight = 1e-2f;
    config.perceptual = 1.0f;
    config.latentCheck = check;
    config.temporal = temporal;
    neural::PhotoEnhancerTraining trainer(vc, config);
    std::vector<std::string> names;
    {
        std::ifstream file(kFixtures + name + "_parameters.txt");
        for (std::string parameter; file >> parameter;)
            names.push_back(parameter);
    }
    EXPECT_EQ(trainer.parameterNames(), names);
    for (const auto& parameter : names)
        trainer.setParameter(parameter, fixture.values("before." + parameter));
    trainer.setBatch(fixture.values("geometry"), fixture.values("coordinates"), fixture.values("classes"),
                     fixture.values("photos"), {});
    trainer.setObjects(fixture.values("objects"));
    trainer.setLatentNoise(fixture.values("epsilon"));
    if (check > 0.0f)
        trainer.setLatentPermutation(fixture.values("permutation"));
    if (temporal > 0.0f)
        trainer.setSecondView(fixture.values("geometry2"), fixture.values("coordinates2"), fixture.values("classes2"),
                              fixture.values("objects2"), fixture.values("flow"), fixture.values("valid"));
    trainer.step();
    EXPECT_LT(relativeError(trainer.image(), fixture.values("image")), 1e-5f);
    EXPECT_NEAR(trainer.loss(), fixture.values("photometric")[0], 1e-5f);
    EXPECT_NEAR(trainer.perceptualLoss(), fixture.values("perceptual")[0], 1e-4f);
    EXPECT_NEAR(trainer.klLoss(), fixture.values("kl")[0], 1e-4f);
    if (check > 0.0f)
        EXPECT_NEAR(trainer.latentCheckLoss(), fixture.values("check")[0], 1e-4f);
    if (temporal > 0.0f)
        EXPECT_NEAR(trainer.temporalLoss(), fixture.values("temporal")[0], 1e-5f);
    for (const auto& parameter : names) {
        const auto before = fixture.values("before." + parameter), after = fixture.values("after." + parameter);
        std::vector<float> actual(after.size());
        trainer.parameter(parameter)->getDataBuffer(0).memcopyTo(actual);
        const float rate = parameter == "texture.table" ? 1e-2f : 1e-4f;
        float worst = 0.0f;
        for (size_t i = 0; i < actual.size(); ++i) {
            // Adam's first step moves a parameter by about the rate unless its gradient is as small as Adam's
            // epsilon; there the step follows the gradient's rounding, so such parameters are left out.
            if (std::abs(after[i] - before[i]) < 0.2f * rate)
                continue;
            worst = std::max(worst, std::abs(actual[i] - after[i]));
        }
        EXPECT_LT(worst, 0.1f * rate) << parameter;
    }
}

} // namespace

TEST(PhotoEnhancerTraining, latentStepMatchesPyTorch) {
    const Fixture fixture("latent_step");
    if (!fixture.exists() || !std::filesystem::exists("build/models/vgg16/layers.txt"))
        GTEST_SKIP() << "run scripts/neural/export_vgg16.py and make_training_fixtures.py first";
    expectLatentStepMatches(fixture, "latent_step", 0.0f);
}

TEST(PhotoEnhancerTraining, latentCheckStepMatchesPyTorch) {
    const Fixture fixture("latent_check_step");
    if (!fixture.exists() || !std::filesystem::exists("build/models/vgg16/layers.txt"))
        GTEST_SKIP() << "run scripts/neural/export_vgg16.py and make_training_fixtures.py first";
    expectLatentStepMatches(fixture, "latent_check_step", 0.5f);
}

TEST(PhotoEnhancerTraining, temporalStepMatchesPyTorch) {
    const Fixture fixture("temporal_step");
    if (!fixture.exists() || !std::filesystem::exists("build/models/vgg16/layers.txt"))
        GTEST_SKIP() << "run scripts/neural/export_vgg16.py and make_training_fixtures.py first";
    expectLatentStepMatches(fixture, "temporal_step", 0.0f, 0.5f);
}

namespace {

// Trains the latent-conditioned renderer with free bits and a PatchGAN (@p conditional, @p scales) from PyTorch's
// renderer and a fresh discriminator on latent_step's batch upsampled @p factor times, and checks the losses.
void expectLatentGanTrains(const Fixture& fixture, bool conditional, uint32_t scales, uint32_t factor) {
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    neural::PhotoEnhancerTrainingConfig config;
    config.batch = 2;
    config.size = 32 * factor;
    config.widths = {32, 64, 128};
    config.gbufferChannels = 4;
    config.texture = neural::HashTextureShape::fromFile(kFixtures + "latent_step_texture.txt");
    config.latentSize = 4;
    config.latentObjects = 1;
    config.klWeight = 1e-2f;
    config.freeBits = 0.5f;
    config.adversarial = 0.1f;
    config.conditionalDiscriminator = conditional;
    config.criticScales = scales;
    neural::PhotoEnhancerTraining trainer(vc, config);
    // The discriminator's first layer sees the G-buffer's 4 channels and the image, or the image only.
    EXPECT_EQ(trainer.parameter("discriminator.discriminators.0.layers.0.0.weight")->getDataElementCount(),
              64u * (conditional ? 7u : 3u) * 16u);
    // PyTorch's renderer, a discriminator from N(0, 0.02) (the PatchGAN's initialisation) with zero biases.
    std::mt19937 rng(3);
    std::normal_distribution<float> normal(0.0f, 0.02f);
    for (const auto& parameter : trainer.parameterNames()) {
        if (parameter.rfind("discriminator.", 0) != 0) {
            trainer.setParameter(parameter, fixture.values("before." + parameter));
            continue;
        }
        std::vector<float> values(trainer.parameter(parameter)->getDataElementCount(), 0.0f);
        if (parameter.find("weight") != std::string::npos)
            for (auto& v : values)
                v = normal(rng);
        trainer.setParameter(parameter, values);
    }
    // The fixture's 32 x 32 batch, every pixel repeated factor x factor.
    const size_t s = config.size;
    const auto upsample = [&](const std::vector<float>& planes) {
        std::vector<float> result(planes.size() * factor * factor);
        for (size_t plane = 0; plane < planes.size() / (32 * 32); ++plane)
            for (size_t y = 0; y < s; ++y)
                for (size_t x = 0; x < s; ++x)
                    result[plane * s * s + y * s + x] = planes[plane * 32 * 32 + (y / factor) * 32 + x / factor];
        return result;
    };
    trainer.setBatch(upsample(fixture.values("geometry")), upsample(fixture.values("coordinates")),
                     upsample(fixture.values("classes")), upsample(fixture.values("photos")), {});
    trainer.setObjects(upsample(fixture.values("objects")));
    trainer.step();
    const float first = trainer.loss();
    // Before the discriminator learns, D is about 0 on both: softplus(0) per real and fake and scale.
    EXPECT_NEAR(trainer.discriminatorLoss(), float(scales) * 2.0f * std::log(2.0f), 0.1f);
    for (int step = 0; step < 30; ++step)
        trainer.step();
    for (const float value : {trainer.loss(), trainer.discriminatorLoss(), trainer.adversarialLoss(),
                              trainer.featureMatchingLoss(), trainer.klLoss()})
        EXPECT_TRUE(std::isfinite(value));
    EXPECT_LT(trainer.loss(), first);
    EXPECT_GE(trainer.klLoss(), 4 * 0.5f - 1e-4f); // 4 numbers, each counted as at least 0.5 nats
}

} // namespace

TEST(PhotoEnhancerTraining, latentGanStepTrains) {
    const Fixture fixture("latent_step");
    if (!fixture.exists())
        GTEST_SKIP() << "run scripts/neural/make_training_fixtures.py first";
    expectLatentGanTrains(fixture, true, 2, 2);
}

TEST(PhotoEnhancerTraining, latentUnconditionalGanStepTrains) {
    const Fixture fixture("latent_step");
    if (!fixture.exists())
        GTEST_SKIP() << "run scripts/neural/make_training_fixtures.py first";
    expectLatentGanTrains(fixture, false, 3, 3);
}
