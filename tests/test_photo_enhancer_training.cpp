// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - generatorStepMatchesPyTorch: one PhotoEnhancerTraining step of a 3-level generator with texture and FiLM
 *   appearance codes, from PyTorch's parameters on PyTorch's batch: the image and the photometric loss equal
 *   PyTorch's, and after the backward pass, clipping and Adam every parameter is within a tenth of its learning
 *   rate of PyTorch's (Adam's first step moves each by about the learning rate, so a wrong gradient shows)
 * - generatorStepWithVggMatchesPyTorch: the same step with the VGG perceptual loss (weight 1) also gives PyTorch's
 *   perceptual loss and updated parameters (skips without the VGG weights)
 * - forwardOnlyRendersTheSameImage: parameters saved from a training and loaded into a forward-only instance render
 *   PyTorch's image of the batch; with every row's code set to one code, the crops render as with that code
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
 * - mlpTextureStepMatchesPyTorch: the same step without VGG and with the latent-conditioned MLP texture (positional
 *   encoding of 3 octaves and the latent G-buffer through 1 x 1 convolutions 16 - 16 - 5): image, losses, every
 *   clipped gradient and every parameter after the step equal latent_renderer.MlpTexture's, the encoder's too (its
 *   gradient also comes through the MLP); without VGG, whose max-pooling makes near-ties of the image amplify
 *   rounding into gradient differences
 * - latentTableStepMatchesPyTorch: the same step without VGG with a learned latent table instead of the encoder (rows
 *   3 and 1 of 5) and the latents only in the MLP texture (the generator's latent channels zero): image, loss, every
 *   clipped gradient (the table's too, with its L2 penalty) and every parameter after the step (the table by its own
 *   unclipped Adam) equal latent_renderer.py's with latent_to_decoder False
 * - untexturedLatentStepTrains: the latent-conditioned renderer without a texture (no levels) has no texture parameter,
 *   takes the G-buffer channels and the latents only, and lowers its photometric loss over 30 steps on one batch
 * - residualMlpTextureStepMatchesPyTorch: the same step with a multi-resolution MLP texture (the first block of 3
 *   octaves and residual blocks of 5 and 7 octaves, their features summed): image, loss, every clipped gradient (of
 *   every block and the table) and every parameter after the step equal latent_renderer.ResidualMlpTexture's
 * - basisTextureStepMatchesPyTorch: the same step with the code-mixed feature texture instead of the MLP texture (6
 *   faces, a base and 4 latent layers of 5 channels at 4 x 4 and 8 x 8 texels): image, loss, every clipped gradient
 *   (the tables' and the latent table's too) and every parameter after the step (the tables by the texture's Adam)
 *   equal latent_renderer.TextureBasis's
 * - basisRegularizedStepMatchesPyTorch: the same step with the feature texture regularised (decay 0.02, smoothness
 *   0.05, each times the level's resolution over the coarsest's): the tables' gradients include
 *   texture_basis_penalty()'s, and every parameter after the step equals PyTorch's
 * - latentsOnlyThroughTheTexture: forward only, two latents give different images with the texture whether or not
 *   the generator takes them; with the texture switched off they still differ when the generator takes them, and
 *   give the same image when only the texture does
 * - latentTableSeparatesTwoAppearances: two samples of the same geometry whose photos are red and blue on the
 *   object, rows 0 and 1 of a 2-row table, latents only in the texture: over 200 steps the loss falls below 30 %,
 *   the rows move apart and the rendered object is red in sample 0 and blue in sample 1
 * - basisTextureSeparatesTwoAppearances: the same with the code-mixed feature texture instead of the MLP texture
 * - latentGanStepTrains: the latent-conditioned renderer with free bits and the 2-scale PatchGAN, from PyTorch's
 *   renderer and a fresh discriminator on its batch upsampled to 64 x 64, starts at the untrained discriminator's loss
 * - latentUnconditionalGanStepTrains: the same with an unconditional discriminator (the image only) of 3 scales
 *   on the batch upsampled 3 times
 * - ganOnlyStepTrains: the latent renderer as a GAN without reconstruction: no encoder parameters, latents given,
 *   the discriminator sees inverse depth and image (4 input channels) with the real photos' own G-buffer, DiffAugment
 *   in the generator step too; it starts at the untrained discriminator's loss, keeps the losses finite and the
 *   adversarial loss alone changes the image
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
#include "training_fixtures.hpp"

using namespace klartraum;
using namespace klartraum::test;

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

namespace {

// One step of the latent-conditioned renderer from latent_renderer.py's parameters on its batch (fixture @p name),
// with the latent check of weight @p check (and the fixture's permutation) and the temporal consistency loss of
// weight @p temporal (and the fixture's second views and flow) if positive: image, losses and every parameter after
// clipping and Adam equal PyTorch's.
// With @p latentCodes the latents come from a learned table (the fixture's rows) instead of the encoder, and with
// @p latentToDecoder false only the texture sees them.
void expectLatentStepMatches(const Fixture& fixture, const std::string& name, float temporal = 0.0f,
                             float perceptual = 1.0f, uint32_t latentCodes = 0, bool latentToDecoder = true,
                             float check = 0.0f) {
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    neural::PhotoEnhancerTrainingConfig config;
    config.batch = 2;
    config.size = 32;
    config.widths = {32, 64, 128};
    config.gbufferChannels = 4;
    config.texture = neural::HashTextureShape::fromFile(kFixtures + name + "_texture.txt");
    config.textureMlp = neural::TextureMlp::fromFile(kFixtures + name + "_texture.txt");
    config.textureBasis = neural::TextureBasis::fromFile(kFixtures + name + "_texture.txt");
    config.latentSize = 4;
    config.latentObjects = 1;
    config.klWeight = 1e-2f;
    config.perceptual = perceptual;
    config.latentCheck = check;
    config.temporal = temporal;
    config.latentCodes = latentCodes;
    config.latentToDecoder = latentToDecoder;
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
                     fixture.values("photos"),
                     fixture.shapes.count("rows") ? fixture.values("rows") : std::vector<float>{});
    trainer.setObjects(fixture.values("objects"));
    if (fixture.shapes.count("epsilon"))
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
        const float rate = parameter.rfind("texture.", 0) == 0 ? 1e-2f : parameter == "latents.weight" ? 5e-3f : 1e-4f;
        float worst = 0.0f;
        for (size_t i = 0; i < actual.size(); ++i) {
            // Adam's first step moves a parameter by about the rate unless its gradient is as small as Adam's
            // epsilon; there the step follows the gradient's rounding, so such parameters are left out.
            if (std::abs(after[i] - before[i]) < 0.2f * rate)
                continue;
            worst = std::max(worst, std::abs(actual[i] - after[i]));
        }
        EXPECT_LT(worst, 0.1f * rate) << parameter;
        // Where the fixture has the clipped gradients, they match too.
        if (fixture.shapes.count("gradient." + parameter)) {
            const auto expected = fixture.values("gradient." + parameter);
            std::vector<float> gradient(expected.size());
            trainer.gradient(parameter)->getDataBuffer(0).memcopyTo(gradient);
            EXPECT_LT(relativeError(gradient, expected), 1e-4f) << parameter;
        }
    }
}

} // namespace

TEST(PhotoEnhancerTraining, latentStepMatchesPyTorch) {
    const Fixture fixture("latent_step");
    if (!fixture.exists() || !std::filesystem::exists("build/models/vgg16/layers.txt"))
        GTEST_SKIP() << "run scripts/neural/export_vgg16.py and make_training_fixtures.py first";
    expectLatentStepMatches(fixture, "latent_step");
}

TEST(PhotoEnhancerTraining, latentCheckStepMatchesPyTorch) {
    const Fixture fixture("latent_check_step");
    if (!fixture.exists() || !std::filesystem::exists("build/models/vgg16/layers.txt"))
        GTEST_SKIP() << "run scripts/neural/export_vgg16.py and make_training_fixtures.py first";
    expectLatentStepMatches(fixture, "latent_check_step", 0.0f, 1.0f, 0, true, 0.5f);
}

TEST(PhotoEnhancerTraining, temporalStepMatchesPyTorch) {
    const Fixture fixture("temporal_step");
    if (!fixture.exists() || !std::filesystem::exists("build/models/vgg16/layers.txt"))
        GTEST_SKIP() << "run scripts/neural/export_vgg16.py and make_training_fixtures.py first";
    expectLatentStepMatches(fixture, "temporal_step", 0.5f);
}

TEST(PhotoEnhancerTraining, mlpTextureStepMatchesPyTorch) {
    const Fixture fixture("mlp_texture_step_l1");
    if (!fixture.exists())
        GTEST_SKIP() << "run scripts/neural/make_training_fixtures.py first";
    expectLatentStepMatches(fixture, "mlp_texture_step_l1", 0.0f, 0.0f);
}

TEST(PhotoEnhancerTraining, latentTableStepMatchesPyTorch) {
    const Fixture fixture("latent_table_step");
    if (!fixture.exists())
        GTEST_SKIP() << "run scripts/neural/make_training_fixtures.py first";
    expectLatentStepMatches(fixture, "latent_table_step", 0.0f, 0.0f, 5, false);
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

TEST(PhotoEnhancerTraining, untexturedLatentStepTrains) {
    const Fixture fixture("latent_step");
    if (!fixture.exists())
        GTEST_SKIP() << "run scripts/neural/make_training_fixtures.py first";
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    neural::PhotoEnhancerTrainingConfig config;
    config.batch = 2;
    config.size = 32;
    config.widths = {32, 64, 128};
    config.gbufferChannels = 4;
    config.texture.levels = 0;
    config.texture.resolutions.clear();
    config.texture.classes = {3};
    config.latentSize = 4;
    config.latentObjects = 1;
    neural::PhotoEnhancerTraining trainer(vc, config);
    const auto names = trainer.parameterNames();
    EXPECT_EQ(std::count(names.begin(), names.end(), "texture.table"), 0);
    // The generator's input: the 4 G-buffer channels and the 4 latents, no texture features.
    EXPECT_EQ(trainer.parameter("generator.down.0.0.weight")->getDataElementCount(), 32u * (4u + 4u) * 9u);
    std::mt19937 rng(5);
    std::uniform_real_distribution<float> uniform(-0.05f, 0.05f);
    for (const auto& name : names) {
        std::vector<float> values(trainer.parameter(name)->getDataElementCount());
        for (auto& v : values)
            v = uniform(rng);
        // The ChannelNorms' scales (layers 1 and 4 of a generator block, 1, 4, 7 of the encoder) start at one.
        const bool norm = (name.find(".1.weight") != std::string::npos || name.find(".4.weight") != std::string::npos ||
                           name.find(".7.weight") != std::string::npos) &&
                          name.rfind("generator.out", 0) != 0;
        if (norm)
            std::fill(values.begin(), values.end(), 1.0f);
        trainer.setParameter(name, values);
    }
    trainer.setBatch(fixture.values("geometry"), fixture.values("coordinates"), fixture.values("classes"),
                     fixture.values("photos"), {});
    trainer.setObjects(fixture.values("objects"));
    trainer.step();
    const float first = trainer.loss();
    for (int step = 0; step < 30; ++step)
        trainer.step();
    EXPECT_TRUE(std::isfinite(trainer.loss()));
    EXPECT_LT(trainer.loss(), first);
}

// A latent renderer with the texture of a latent table fixture (the MLP texture or the feature texture).
neural::PhotoEnhancerTrainingConfig latentTextureConfig(const std::string& name) {
    neural::PhotoEnhancerTrainingConfig config;
    config.batch = 2;
    config.size = 32;
    config.widths = {32, 64, 128};
    config.gbufferChannels = 4;
    config.texture = neural::HashTextureShape::fromFile(kFixtures + name + "_texture.txt");
    config.textureMlp = neural::TextureMlp::fromFile(kFixtures + name + "_texture.txt");
    config.textureBasis = neural::TextureBasis::fromFile(kFixtures + name + "_texture.txt");
    config.latentSize = 4;
    config.latentObjects = 1;
    return config;
}

// Small random parameters (ChannelNorm scales one, the output layer larger so that the image follows its input).

void randomiseParameters(neural::PhotoEnhancerTraining& trainer, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uniform(-0.2f, 0.2f);
    for (const auto& name : trainer.parameterNames()) {
        std::vector<float> values(trainer.parameter(name)->getDataElementCount());
        for (auto& v : values)
            v = uniform(rng);
        const bool block = name.rfind("generator.down.", 0) == 0 || name.rfind("generator.up.", 0) == 0;
        const bool norm =
            block && (name.find(".1.weight") != std::string::npos || name.find(".4.weight") != std::string::npos);
        if (norm)
            std::fill(values.begin(), values.end(), 1.0f);
        trainer.setParameter(name, values);
    }
}

TEST(PhotoEnhancerTraining, residualMlpTextureStepMatchesPyTorch) {
    const Fixture fixture("residual_mlp_step");
    if (!fixture.exists())
        GTEST_SKIP() << "run scripts/neural/make_training_fixtures.py first";
    expectLatentStepMatches(fixture, "residual_mlp_step", 0.0f, 0.0f, 5, false);
}

TEST(PhotoEnhancerTraining, basisTextureStepMatchesPyTorch) {
    const Fixture fixture("basis_texture_step");
    if (!fixture.exists())
        GTEST_SKIP() << "run scripts/neural/make_training_fixtures.py first";
    expectLatentStepMatches(fixture, "basis_texture_step", 0.0f, 0.0f, 5, false);
}

TEST(PhotoEnhancerTraining, basisRegularizedStepMatchesPyTorch) {
    const Fixture fixture("basis_regularized_step");
    if (!fixture.exists())
        GTEST_SKIP() << "run scripts/neural/make_training_fixtures.py first";
    expectLatentStepMatches(fixture, "basis_regularized_step", 0.0f, 0.0f, 5, false);
}

TEST(PhotoEnhancerTraining, latentsOnlyThroughTheTexture) {
    const Fixture fixture("latent_table_step");
    if (!fixture.exists())
        GTEST_SKIP() << "run scripts/neural/make_training_fixtures.py first";
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    // Two latents per configuration: the images with the texture, then with it switched off.
    const std::vector<float> first{0.9f, -0.4f, 0.3f, 1.2f, 0.9f, -0.4f, 0.3f, 1.2f};
    const std::vector<float> second{-1.1f, 0.6f, -0.8f, 0.2f, -1.1f, 0.6f, -0.8f, 0.2f};
    const auto render = [&](bool latentToDecoder, bool textureOff, const std::vector<float>& latents) {
        auto config = latentTextureConfig("latent_table_step");
        config.forwardOnly = true;
        config.latentToDecoder = latentToDecoder;
        config.textureOff = textureOff;
        neural::PhotoEnhancerTraining trainer(vc, config);
        randomiseParameters(trainer, 3);
        trainer.setBatch(fixture.values("geometry"), fixture.values("coordinates"), fixture.values("classes"),
                         fixture.values("photos"), {});
        trainer.setObjects(fixture.values("objects"));
        trainer.setLatents(latents);
        trainer.step();
        return trainer.image();
    };
    // The texture carries the latents in both configurations.
    EXPECT_GT(relativeError(render(false, false, first), render(false, false, second)), 1e-3f);
    EXPECT_GT(relativeError(render(true, false, first), render(true, false, second)), 1e-3f);
    // Without the texture, only a generator that takes the latents still changes with them.
    EXPECT_GT(relativeError(render(true, true, first), render(true, true, second)), 1e-3f);
    EXPECT_LT(relativeError(render(false, true, first), render(false, true, second)), 1e-6f);
}

// Trains rows 0 and 1 of a 2-row latent table, latents only in the texture of the fixture `name`, on two samples of the
// same geometry whose photos are red and blue on the object.
void expectTwoAppearancesSeparate(const std::string& name) {
    const Fixture fixture(name);
    if (!fixture.exists())
        GTEST_SKIP() << "run scripts/neural/make_training_fixtures.py first";
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    auto config = latentTextureConfig(name);
    config.latentCodes = 2;
    config.latentToDecoder = false;
    config.networkAdam.learningRate = 2e-3f;
    neural::PhotoEnhancerTraining trainer(vc, config);
    randomiseParameters(trainer, 4);
    trainer.setParameter("latents.weight", {0.05f, -0.03f, 0.02f, 0.01f, -0.04f, 0.02f, -0.01f, 0.03f});
    // Both samples have the same geometry; on the object, sample 0's photo is red and sample 1's blue.
    const size_t pixels = 32 * 32;
    auto geometry = fixture.values("geometry"), coordinates = fixture.values("coordinates");
    auto classes = fixture.values("classes"), objects = fixture.values("objects");
    std::copy(geometry.begin(), geometry.begin() + 4 * pixels, geometry.begin() + 4 * pixels);
    std::copy(coordinates.begin(), coordinates.begin() + 3 * pixels, coordinates.begin() + 3 * pixels);
    std::copy(classes.begin(), classes.begin() + pixels, classes.begin() + pixels);
    std::copy(objects.begin(), objects.begin() + pixels, objects.begin() + pixels);
    std::vector<float> photos(2 * 3 * pixels, 0.5f);
    for (size_t p = 0; p < pixels; ++p)
        if (objects[p] > 0.0f)
            for (size_t c = 0; c < 3; ++c) {
                photos[c * pixels + p] = c == 0 ? 0.9f : 0.1f;
                photos[(3 + c) * pixels + p] = c == 2 ? 0.9f : 0.1f;
            }
    trainer.setBatch(geometry, coordinates, classes, photos, {0.0f, 1.0f});
    trainer.setObjects(objects);
    trainer.step();
    const float first = trainer.loss();
    for (int step = 0; step < 200; ++step)
        trainer.step();
    EXPECT_LT(trainer.loss(), 0.3f * first);
    // The rows moved apart, and the image's object is red in sample 0 and blue in sample 1.
    std::vector<float> table(8);
    trainer.parameter("latents.weight")->getDataBuffer(0).memcopyTo(table);
    float distance = 0.0f;
    for (size_t i = 0; i < 4; ++i)
        distance += (table[i] - table[4 + i]) * (table[i] - table[4 + i]);
    EXPECT_GT(std::sqrt(distance), 0.2f);
    const auto image = trainer.image();
    float red[2] = {0, 0}, blue[2] = {0, 0};
    for (size_t p = 0; p < pixels; ++p)
        if (objects[p] > 0.0f)
            for (size_t n = 0; n < 2; ++n) {
                red[n] += image[(3 * n + 0) * pixels + p];
                blue[n] += image[(3 * n + 2) * pixels + p];
            }
    EXPECT_GT(red[0], blue[0]);
    EXPECT_GT(blue[1], red[1]);
}

TEST(PhotoEnhancerTraining, latentTableSeparatesTwoAppearances) { expectTwoAppearancesSeparate("latent_table_step"); }

TEST(PhotoEnhancerTraining, basisTextureSeparatesTwoAppearances) { expectTwoAppearancesSeparate("basis_texture_step"); }

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

TEST(PhotoEnhancerTraining, ganOnlyStepTrains) {
    const Fixture fixture("latent_step");
    if (!fixture.exists())
        GTEST_SKIP() << "run scripts/neural/make_training_fixtures.py first";
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    neural::PhotoEnhancerTrainingConfig config;
    config.batch = 2;
    config.size = 64;
    config.widths = {32, 64, 128};
    config.gbufferChannels = 4;
    config.texture = neural::HashTextureShape::fromFile(kFixtures + "latent_step_texture.txt");
    config.latentSize = 4;
    config.latentObjects = 1;
    config.randomLatents = true;
    config.l1 = config.colour = 0.0f;
    config.adversarial = 1.0f;
    config.featureMatching = 0.0f;
    config.criticGBufferChannels = 1;
    config.separateRealGBuffer = true;
    config.augmentGeneratorStep = true;
    neural::PhotoEnhancerTraining trainer(vc, config);
    const auto names = trainer.parameterNames();
    EXPECT_EQ(
        std::count_if(names.begin(), names.end(), [](const std::string& n) { return n.rfind("encoder.", 0) == 0; }), 0);
    // The discriminator sees the inverse depth and the image.
    EXPECT_EQ(trainer.parameter("discriminator.discriminators.0.layers.0.0.weight")->getDataElementCount(),
              64u * (1u + 3u) * 16u);
    std::mt19937 rng(3);
    std::normal_distribution<float> normal(0.0f, 0.02f);
    for (const auto& parameter : names) {
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
    const auto upsample = [](const std::vector<float>& planes) {
        std::vector<float> result(planes.size() * 4);
        for (size_t plane = 0; plane < planes.size() / (32 * 32); ++plane)
            for (size_t y = 0; y < 64; ++y)
                for (size_t x = 0; x < 64; ++x)
                    result[plane * 64 * 64 + y * 64 + x] = planes[plane * 32 * 32 + (y / 2) * 32 + x / 2];
        return result;
    };
    // The generated views and the real photos come with different G-buffers (here: the batch's samples swapped).
    const auto geometry = upsample(fixture.values("geometry"));
    std::vector<float> swapped(geometry.begin() + geometry.size() / 2, geometry.end());
    swapped.insert(swapped.end(), geometry.begin(), geometry.begin() + geometry.size() / 2);
    trainer.setBatch(geometry, upsample(fixture.values("coordinates")), upsample(fixture.values("classes")),
                     upsample(fixture.values("photos")), {});
    trainer.setObjects(upsample(fixture.values("objects")));
    trainer.setRealGBuffer(swapped);
    std::normal_distribution<float> unit(0.0f, 1.0f);
    std::vector<float> latents(2 * 4);
    for (auto& v : latents)
        v = unit(rng);
    trainer.setLatents(latents);
    trainer.step();
    const auto first = trainer.image();
    EXPECT_NEAR(trainer.discriminatorLoss(), 2.0f * 2.0f * std::log(2.0f), 0.1f);
    EXPECT_EQ(trainer.klLoss(), 0.0f);
    for (int step = 0; step < 20; ++step)
        trainer.step();
    for (const float value : {trainer.discriminatorLoss(), trainer.adversarialLoss()})
        EXPECT_TRUE(std::isfinite(value));
    // Only the adversarial loss trains the generator, so its image changes.
    EXPECT_GT(relativeError(trainer.image(), first), 1e-3f);
}
