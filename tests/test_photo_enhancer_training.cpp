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
 * - forwardOnlyRendersTheSameImage: parameters saved from a training and loaded into a forward-only instance render
 *   PyTorch's image of the batch; with every row's code set to one code, the crops render as with that code
 * - maxPoolMatchesPyTorch: a 2 x 2 max-pool and its input gradient equal F.max_pool2d's
 * - vggLossMatchesPyTorch: the perceptual loss of an image against a target through VGG-16 up to relu4_3 and its
 *   gradient (added to a zeroed image gradient) equal train_photo_enhancer.Vgg's (skips without
 *   scripts/neural/export_vgg16.py's weights)
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

TEST(PhotoEnhancerTraining, generatorStepMatchesPyTorch) {
    const Fixture fixture("generator_step");
    if (!fixture.exists())
        GTEST_SKIP() << "run scripts/neural/make_training_fixtures.py first";
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    neural::PhotoEnhancerTrainingConfig config;
    config.batch = 2;
    config.size = 32;
    config.widths = {32, 64, 128};
    config.texture = neural::HashTextureShape::fromFile(kFixtures + "generator_step_texture.txt");
    config.appearances = 5;
    config.appearanceSize = 8;
    neural::PhotoEnhancerTraining trainer(vc, config);
    std::vector<std::string> names;
    {
        std::ifstream file(kFixtures + "generator_step_parameters.txt");
        for (std::string name; file >> name;)
            names.push_back(name);
    }
    EXPECT_EQ(trainer.parameterNames(), names);
    for (const auto& name : names)
        trainer.setParameter(name, fixture.values("before." + name));
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
    EXPECT_NEAR(trainer.loss(), fixture.values("loss")[0] - 1e-2f * penalty / 2.0f, 1e-5f);
    for (const auto& name : names) {
        const auto before = fixture.values("before." + name), after = fixture.values("after." + name);
        std::vector<float> actual(before.size());
        trainer.parameter(name)->getDataBuffer(0).memcopyTo(actual);
        const float rate = name == "texture.table" ? 1e-2f : name == "codes.weight" ? 5e-3f : 1e-4f;
        float worst = 0.0f;
        for (size_t i = 0; i < actual.size(); ++i)
            worst = std::max(worst, std::abs(actual[i] - after[i]));
        EXPECT_LT(worst, 0.1f * rate) << name;
    }
}

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
