// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - activationBackwardMatchesPyTorch: the ELU and leaky ReLU (0.2) input gradients, and the leaky ReLU itself, equal
 *   PyTorch's
 * - channelCopyConcatsAndSplits: copying two tensors' channels to their offsets concatenates them, copying them back
 *   out recovers them, and accumulating adds
 * - linearMatchesPyTorch: a linear layer's output (with an offset) and its input (also accumulating), weight and
 *   bias gradients equal nn.Linear's
 * - convBackwardMatchesPyTorch: the input, weight and bias gradients of 2D convolutions (3x3 with stride 1 and 2,
 *   1x1, 4x4 with stride 2) equal PyTorch autograd's for the same inputs and upstream gradient
 * - channelNormMatchesPyTorch: ChannelNorm's output and its input, scale and shift gradients equal PyTorch's
 *   (train_photo_enhancer.ChannelNorm)
 * - upsampleBackwardMatchesPyTorch: the input gradient of a bilinear upsampling by two equals PyTorch's
 *   (F.interpolate, align_corners=False)
 * - maxPoolMatchesPyTorch: a 2 x 2 max-pool and its input gradient equal F.max_pool2d's
 * - averagePoolMatchesMeans: a 2 x 2 average pool gives the mean of every 2 x 2 block, and its backward pass spreads a
 *   quarter of each output gradient onto the block's pixels, written or added to what is there
 * - embeddingPicksAndScattersRows: rows picked by index (one twice) come out in batch order; the table gradient adds
 *   the picked rows' gradients plus the L2 penalty's, and is zero for rows nobody picked
 * - filmMatchesPyTorch: FiLM's output and its input, scale and shift gradients equal PyTorch's
 * - clipGradientMatchesPyTorch: clipping two tensors to a global norm of 1 equals clip_grad_norm_
 * - emaCopiesThenAverages: during the warm-up the average copies the parameters, after it moves by the rate
 * - photometricLossMatchesPyTorch: the gradient of 0.5 L1 + 5 L1 of 8 x 8 block means equals PyTorch's, and the sum
 *   of the loss shares is the loss
 * - vggLossMatchesPyTorch: the perceptual loss of an image against a target through VGG-16 up to relu4_3 and its
 *   gradient (added to a zeroed image gradient) equal train_photo_enhancer.Vgg's (skips without
 *   scripts/neural/export_vgg16.py's weights)
 * - softplusLossMatchesMean: the GAN loss over a range of logits gives each logit its share weight x softplus(sign x) /
 *   count (also for large logits) and the gradient weight x sign x sigmoid(sign x) / count, and leaves the logits
 *   outside the range alone
 * - augmentPairBackwardMatchesPyTorch: DiffAugment of discriminator pairs (brightness, contrast, rolls both ways,
 *   cutouts inside and at the border) and its input gradient equal augment_with()'s and autograd's
 * - objectMeanMatchesPyTorch: the mean over each object's pixels (an object missing in one sample) and the pixel
 *   counts, their backward pass (broadcast divided by the counts), the latent G-buffer (broadcast) and its backward
 *   pass (sum over the object's pixels) equal latent_renderer.object_mean()'s and splat()'s
 * - vaeMatchesPyTorch: the VAE's latents and KL divergence and the statistics' gradient equal PyTorch's
 * - vaeFreeBitsMatchesPyTorch: with 0.3 free bits, the KL term (each number counted as at least 0.3 nats) and the
 *   statistics' gradient (no KL gradient for numbers below) equal PyTorch's
 * - flowWarpLossMatchesPyTorch: the temporal consistency loss (the first render warped by a flow partly beyond the
 *   image, bilinear and clamped, against the second on a mask) and its gradient for the second render equal
 *   latent_renderer.flow_warp_loss()'s (grid_sample)
 * - hashTextureBackwardMatchesPyTorch: for a batch of two images, the texture's features and its table gradient (the
 *   atomic additions into a zeroed gradient) equal PyTorch's (hash_texture.SurfaceTexture)
 * - positionalEncodingMatchesPyTorch: the positional encoding of surface coordinates (sine and cosine of 4 octaves per
 *   coordinate, then the coordinates; zero off the texture's classes) equals latent_renderer.positional_encoding()'s
 * - textureBasisMatchesPyTorch: the code-mixed feature texture (6 faces, a base and 2 latent layers of 4 channels at
 *   5 x 5 texels; coordinates partly beyond the edges, classes 3 and 1 textured) writes latent_renderer.
 *   sample_texture_basis()'s features at its channels of a wider output and leaves the others alone; its backward
 *   pass gives autograd's table and latent gradients
 * - textureBasisRegularizeMatchesPyTorch: the feature texture's regularisation (decay 0.3 x squared texels + smoothness
 *   0.7 x squared differences of neighbouring texels within each plane) adds latent_renderer.texture_basis_penalty()'s
 *   gradient to the table's gradient
 *
 * The references come from scripts/neural/make_element_fixtures.py (build/TestingOutput/training_fixtures/); the
 * tests skip without them.
 **/

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/tensorelement.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/layers/training.hpp"
#include "klartraum/neural/vgg_loss.hpp"
#include "klartraum/neural/hash_texture.hpp"
#include "training_fixtures.hpp"

using namespace klartraum;
using namespace klartraum::test;

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

TEST(TrainingElements, activationBackwardMatchesPyTorch) {
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    for (const auto& [name, activation] :
         {std::pair{"elu", training::Activation::Elu}, std::pair{"leaky_relu", training::Activation::LeakyRelu}}) {
        const Fixture fixture(name);
        if (!fixture.exists())
            GTEST_SKIP() << "run scripts/neural/make_element_fixtures.py first";
        const auto shape = fixture.shapes.at("x");
        checkElement(vc, fixture, training::activationBackward(vc, activation, shape),
                     {{"x", shape}, {"dy", shape}, {"dx", shape}}, 2);
        if (activation == training::Activation::LeakyRelu)
            checkElement(vc, fixture, training::leakyRelu(vc, shape), {{"x", shape}, {"y", shape}}, 1);
    }
}

TEST(TrainingElements, channelCopyConcatsAndSplits) {
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

TEST(TrainingElements, linearMatchesPyTorch) {
    const Fixture fixture("linear");
    if (!fixture.exists())
        GTEST_SKIP() << "run scripts/neural/make_element_fixtures.py first";
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

TEST(TrainingElements, convBackwardMatchesPyTorch) {
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
            GTEST_SKIP() << "run scripts/neural/make_element_fixtures.py first";
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

TEST(TrainingElements, channelNormMatchesPyTorch) {
    const Fixture fixture("channel_norm");
    if (!fixture.exists())
        GTEST_SKIP() << "run scripts/neural/make_element_fixtures.py first";
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

TEST(TrainingElements, upsampleBackwardMatchesPyTorch) {
    const Fixture fixture("upsample2x");
    if (!fixture.exists())
        GTEST_SKIP() << "run scripts/neural/make_element_fixtures.py first";
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    const auto shape = fixture.shapes.at("x");
    checkElement(vc, fixture, training::upsample2xBackward(vc, shape), {{"dy", fixture.shapes.at("y")}, {"dx", shape}},
                 1);
}

TEST(TrainingElements, maxPoolMatchesPyTorch) {
    const Fixture fixture("max_pool");
    if (!fixture.exists())
        GTEST_SKIP() << "run scripts/neural/make_element_fixtures.py first";
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

TEST(TrainingElements, averagePoolMatchesMeans) {
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    const std::vector<uint32_t> shape{2, 3, 4, 6}, pooled{2, 3, 2, 3};
    auto x = tensor(vc, shape), y = tensor(vc, pooled), dy = tensor(vc, pooled), dx = tensor(vc, shape);
    auto dxAdded = tensor(vc, shape);
    auto forward = training::averagePool(vc, shape);
    forward->setInput(x, 0);
    forward->setInput(y, 1);
    auto backward = training::averagePoolBackward(vc, shape);
    backward->setInput(dy, 0);
    backward->setInput(dx, 1);
    backward->addDependency(forward);
    auto added = training::averagePoolBackward(vc, shape, true);
    added->setInput(dy, 0);
    added->setInput(dxAdded, 1);
    added->addDependency(backward);
    ComputeGraph graph(vc, 1);
    graph.compileFrom(added);
    std::mt19937 rng(3);
    std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
    std::vector<float> xs(2 * 3 * 4 * 6), dys(2 * 3 * 2 * 3);
    for (auto& v : xs)
        v = uniform(rng);
    for (auto& v : dys)
        v = uniform(rng);
    x->getDataBuffer(0).memcopyFrom(xs);
    dy->getDataBuffer(0).memcopyFrom(dys);
    dxAdded->getDataBuffer(0).memcopyFrom(std::vector<float>(xs.size(), 1.0f));
    graph.submitAndWait(vc.getGraphicsQueue(), 0);
    const auto ys = read(y), dxs = read(dx), dxsAdded = read(dxAdded);
    for (uint32_t plane = 0; plane < 6; ++plane)
        for (uint32_t oy = 0; oy < 2; ++oy)
            for (uint32_t ox = 0; ox < 3; ++ox) {
                float mean = 0.0f;
                for (uint32_t k = 0; k < 4; ++k)
                    mean += 0.25f * xs[(plane * 4 + 2 * oy + k / 2) * 6 + 2 * ox + k % 2];
                const size_t o = (plane * 2 + oy) * 3 + ox;
                EXPECT_NEAR(ys[o], mean, 1e-6f);
                for (uint32_t k = 0; k < 4; ++k) {
                    const size_t i = (plane * 4 + 2 * oy + k / 2) * 6 + 2 * ox + k % 2;
                    EXPECT_NEAR(dxs[i], 0.25f * dys[o], 1e-6f);
                    EXPECT_NEAR(dxsAdded[i], 1.0f + 0.25f * dys[o], 1e-6f);
                }
            }
}

TEST(TrainingElements, embeddingPicksAndScattersRows) {
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

TEST(TrainingElements, filmMatchesPyTorch) {
    const Fixture fixture("film");
    if (!fixture.exists())
        GTEST_SKIP() << "run scripts/neural/make_element_fixtures.py first";
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

TEST(TrainingElements, clipGradientMatchesPyTorch) {
    const Fixture fixture("clip");
    if (!fixture.exists())
        GTEST_SKIP() << "run scripts/neural/make_element_fixtures.py first";
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

TEST(TrainingElements, emaCopiesThenAverages) {
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

TEST(TrainingElements, photometricLossMatchesPyTorch) {
    const Fixture fixture("photometric");
    if (!fixture.exists())
        GTEST_SKIP() << "run scripts/neural/make_element_fixtures.py first";
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

TEST(TrainingElements, vggLossMatchesPyTorch) {
    const Fixture fixture("vgg");
    const std::string weights = "build/models/vgg16";
    if (!fixture.exists() || !std::filesystem::exists(weights + "/layers.txt"))
        GTEST_SKIP() << "run scripts/neural/export_vgg16.py and make_element_fixtures.py first";
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

TEST(TrainingElements, softplusLossMatchesMean) {
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    const std::vector<float> logits{9.0f, -1.5f, 0.0f, 0.7f, 40.0f, -40.0f, 2.0f};
    const uint32_t offset = 1, count = 5;
    const float weight = 0.6f;
    for (const float sign : {-1.0f, 1.0f}) {
        auto x = tensor(vc, {7}), gradient = tensor(vc, {7}), shares = tensor(vc, {7});
        auto loss = training::softplusLoss(vc, offset, count, sign, weight);
        loss->setInput(x, 0);
        loss->setInput(gradient, 1);
        loss->setInput(shares, 2);
        ComputeGraph graph(vc, 1);
        graph.compileFrom(loss);
        x->getDataBuffer(0).memcopyFrom(logits);
        gradient->getDataBuffer(0).memcopyFrom(std::vector<float>(7, -7.0f));
        shares->getDataBuffer(0).memcopyFrom(std::vector<float>(7, -7.0f));
        graph.submitAndWait(vc.getGraphicsQueue(), 0);
        const auto g = read(gradient), s = read(shares);
        for (uint32_t i = 0; i < 7; ++i) {
            if (i < offset || i >= offset + count) {
                EXPECT_EQ(g[i], -7.0f);
                EXPECT_EQ(s[i], -7.0f);
                continue;
            }
            const double z = double(sign) * logits[i];
            const double softplus = std::max(z, 0.0) + std::log1p(std::exp(-std::abs(z)));
            EXPECT_NEAR(s[i], weight * softplus / count, 1e-5) << i;
            EXPECT_NEAR(g[i], weight * sign / (1.0 + std::exp(-z)) / count, 1e-6) << i;
        }
    }
}

TEST(TrainingElements, augmentPairBackwardMatchesPyTorch) {
    const Fixture fixture("augment_backward");
    if (!fixture.exists())
        GTEST_SKIP() << "run scripts/neural/make_element_fixtures.py first";
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    const auto shape = fixture.shapes.at("pair");
    auto pair = tensor(vc, shape), numbers = tensor(vc, fixture.shapes.at("parameters"));
    auto augmented = tensor(vc, shape), daugmented = tensor(vc, shape), dpair = tensor(vc, shape);
    auto forward = training::augmentPair(vc, shape);
    forward->setInput(pair, 0);
    forward->setInput(numbers, 1);
    forward->setInput(augmented, 2);
    auto backward = training::augmentPairBackward(vc, shape);
    backward->setInput(numbers, 0);
    backward->setInput(daugmented, 1);
    backward->setInput(dpair, 2);
    backward->addDependency(forward);
    ComputeGraph graph(vc, 1);
    graph.compileFrom(backward);
    pair->getDataBuffer(0).memcopyFrom(fixture.values("pair"));
    numbers->getDataBuffer(0).memcopyFrom(fixture.values("parameters"));
    daugmented->getDataBuffer(0).memcopyFrom(fixture.values("daugmented"));
    graph.submitAndWait(vc.getGraphicsQueue(), 0);
    EXPECT_LT(relativeError(read(augmented), fixture.values("augmented")), 1e-5f);
    EXPECT_LT(relativeError(read(dpair), fixture.values("dpair")), 1e-5f);
}

TEST(TrainingElements, objectMeanMatchesPyTorch) {
    const Fixture fixture("object_mean");
    if (!fixture.exists())
        GTEST_SKIP() << "run scripts/neural/make_element_fixtures.py first";
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

TEST(TrainingElements, vaeMatchesPyTorch) {
    const Fixture fixture("vae");
    if (!fixture.exists())
        GTEST_SKIP() << "run scripts/neural/make_element_fixtures.py first";
    expectVaeMatches(fixture, fixture, 0.0f);
}

TEST(TrainingElements, vaeFreeBitsMatchesPyTorch) {
    const Fixture fixture("vae"), expected("vae_free_bits");
    if (!fixture.exists() || !expected.exists())
        GTEST_SKIP() << "run scripts/neural/make_element_fixtures.py first";
    expectVaeMatches(fixture, expected, 0.3f);
}

TEST(TrainingElements, flowWarpLossMatchesPyTorch) {
    const Fixture fixture("flow_warp");
    if (!fixture.exists())
        GTEST_SKIP() << "run scripts/neural/make_element_fixtures.py first";
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

TEST(TrainingElements, hashTextureBackwardMatchesPyTorch) {
    const Fixture fixture("hash_texture");
    if (!fixture.exists())
        GTEST_SKIP() << "run scripts/neural/make_element_fixtures.py first";
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

TEST(TrainingElements, positionalEncodingMatchesPyTorch) {
    const Fixture fixture("positional_encoding");
    if (!fixture.exists())
        GTEST_SKIP() << "run scripts/neural/make_element_fixtures.py first";
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    const auto shape = fixture.shapes.at("coordinates");
    auto coordinates = tensor(vc, shape), classes = tensor(vc, fixture.shapes.at("classes"));
    auto encoding = tensor(vc, fixture.shapes.at("encoding"));
    auto encode = training::positionalEncoding(vc, shape[0], shape[2] * shape[3], 4, {3, 1});
    encode->setInput(coordinates, 0);
    encode->setInput(classes, 1);
    encode->setInput(encoding, 2);
    ComputeGraph graph(vc, 1);
    graph.compileFrom(encode);
    coordinates->getDataBuffer(0).memcopyFrom(fixture.values("coordinates"));
    classes->getDataBuffer(0).memcopyFrom(fixture.values("classes"));
    graph.submitAndWait(vc.getGraphicsQueue(), 0);
    const auto actual = read(encoding), expected = fixture.values("encoding");
    float worst = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i)
        worst = std::max(worst, std::abs(actual[i] - expected[i]));
    EXPECT_LT(worst, 1e-5f);
}

TEST(TrainingElements, textureBasisMatchesPyTorch) {
    const Fixture fixture("texture_basis");
    if (!fixture.exists())
        GTEST_SKIP() << "run scripts/neural/make_element_fixtures.py first";
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    // 2 samples of 5 x 7 pixels; the features at channels 2..5 of a 6-channel output.
    const uint32_t n = 2, pixels = 35, channels = 4, outputChannels = 6;
    training::TextureBasisShape shape;
    shape.batch = n;
    shape.pixels = pixels;
    shape.faces = 6;
    shape.layers = 3;
    shape.channels = channels;
    shape.resolution = 5;
    shape.classes = {3, 1};
    shape.outputChannels = outputChannels;
    shape.outputOffset = 2;
    auto coordinates = tensor(vc, {n, 3, 5, 7}), classes = tensor(vc, {n, 5, 7});
    auto table = tensor(vc, {uint32_t(shape.tableSize())}), latents = tensor(vc, {n, 2, 5, 7});
    auto output = tensor(vc, {n, outputChannels, 5, 7}), outputGradient = tensor(vc, {n, outputChannels, 5, 7});
    auto tableGradient = tensor(vc, {uint32_t(shape.tableSize())}), latentGradient = tensor(vc, {n, 2, 5, 7});
    auto forward = training::textureBasis(vc, shape);
    forward->setInput(coordinates, 0);
    forward->setInput(classes, 1);
    forward->setInput(table, 2);
    forward->setInput(latents, 3);
    forward->setInput(output, 4);
    auto backward = training::textureBasisBackward(vc, shape);
    backward->setInput(coordinates, 0);
    backward->setInput(classes, 1);
    backward->setInput(table, 2);
    backward->setInput(latents, 3);
    backward->setInput(outputGradient, 4);
    backward->setInput(tableGradient, 5);
    backward->setInput(latentGradient, 6);
    backward->addDependency(forward);
    ComputeGraph graph(vc, 1);
    graph.compileFrom(backward);
    coordinates->getDataBuffer(0).memcopyFrom(fixture.values("coordinates"));
    classes->getDataBuffer(0).memcopyFrom(fixture.values("classes"));
    table->getDataBuffer(0).memcopyFrom(fixture.values("table"));
    latents->getDataBuffer(0).memcopyFrom(fixture.values("latents"));
    // The other channels of the output keep their values, and their gradient plays no part.
    std::vector<float> before(n * outputChannels * pixels, 7.0f), gradient(n * outputChannels * pixels, 100.0f);
    const auto upstream = fixture.values("upstream");
    for (uint32_t k = 0; k < n; ++k)
        for (uint32_t c = 0; c < channels; ++c)
            for (uint32_t p = 0; p < pixels; ++p)
                gradient[(k * outputChannels + 2 + c) * pixels + p] = upstream[(k * channels + c) * pixels + p];
    output->getDataBuffer(0).memcopyFrom(before);
    outputGradient->getDataBuffer(0).memcopyFrom(gradient);
    tableGradient->getDataBuffer(0).memcopyFrom(std::vector<float>(shape.tableSize(), 0.0f));
    graph.submitAndWait(vc.getGraphicsQueue(), 0);
    const auto actual = read(output), features = fixture.values("features");
    float worst = 0.0f;
    for (uint32_t k = 0; k < n; ++k)
        for (uint32_t c = 0; c < outputChannels; ++c)
            for (uint32_t p = 0; p < pixels; ++p) {
                const float expected = c < 2 ? 7.0f : features[(k * channels + c - 2) * pixels + p];
                worst = std::max(worst, std::abs(actual[(k * outputChannels + c) * pixels + p] - expected));
            }
    EXPECT_LT(worst, 1e-5f);
    EXPECT_LT(relativeError(read(tableGradient), fixture.values("dtable")), 1e-5f);
    EXPECT_LT(relativeError(read(latentGradient), fixture.values("dlatents")), 1e-5f);
}

TEST(TrainingElements, textureBasisRegularizeMatchesPyTorch) {
    const Fixture fixture("texture_basis_penalty");
    if (!fixture.exists())
        GTEST_SKIP() << "run scripts/neural/make_element_fixtures.py first";
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    training::TextureBasisShape shape;
    shape.faces = 6;
    shape.layers = 3;
    shape.channels = 4;
    shape.resolution = 5;
    auto table = tensor(vc, {uint32_t(shape.tableSize())}), gradient = tensor(vc, {uint32_t(shape.tableSize())});
    auto regularize = training::textureBasisRegularize(vc, shape, 0.3f, 0.7f);
    regularize->setInput(table, 0);
    regularize->setInput(gradient, 1);
    ComputeGraph graph(vc, 1);
    graph.compileFrom(regularize);
    table->getDataBuffer(0).memcopyFrom(fixture.values("table"));
    // The penalty's gradient adds to what is there.
    gradient->getDataBuffer(0).memcopyFrom(std::vector<float>(shape.tableSize(), 1.0f));
    graph.submitAndWait(vc.getGraphicsQueue(), 0);
    auto expected = fixture.values("dtable");
    for (auto& v : expected)
        v += 1.0f;
    EXPECT_LT(relativeError(read(gradient), expected), 1e-5f);
}
