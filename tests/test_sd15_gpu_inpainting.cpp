// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - lcmInitMatchesCpu: LCM init writes known = encoded * scale, the start latent of sampleInpainted() and the first
 *   timestep
 * - lcmStepMatchesScheduler: a middle and the last LCM step equal LcmScheduler::step() followed by the latent
 *   blending of sampleInpainted(); the middle step sets the next timestep, the last one writes the decoder input
 * - chainMatchesSampleInpainted: init and all steps chained with a stand-in denoiser (prediction = sample * factor)
 *   give the latents of sampleInpainted() with the same predictor and the exported LCM schedule
 **/

#include <cstring>
#include <memory>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/graphchain.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/sd15/gpu_inpainting.hpp"
#include "klartraum/sd15/latent_inpainting.hpp"
#include "klartraum/sd15/lcm_scheduler.hpp"

using namespace klartraum;

namespace {

constexpr uint32_t kChannels = 4;
constexpr uint32_t kPlane = 16 * 16;
constexpr uint32_t kCount = kChannels * kPlane;
constexpr float kScale = 0.18215f;

using Floats = BufferElementSinglePath<VulkanBuffer<float>>;
using Words = BufferElementSinglePath<VulkanBuffer<uint32_t>>;

LcmScheduler schedule() { return LcmScheduler::load("data/onnx/sd15_controlnet_512/lcm_schedule_f32.bin"); }

std::vector<float> random(size_t count, uint32_t seed, float low = -1.0f, float high = 1.0f) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(low, high);
    std::vector<float> values(count);
    for (auto& value : values) {
        value = distribution(generator);
    }
    return values;
}

// A mask with generated (1), kept (0) and half-generated (0.5) latent pixels.
std::vector<float> mixedMask() {
    std::vector<float> mask(kPlane);
    for (uint32_t i = 0; i < kPlane; ++i) {
        mask[i] = i % 3 == 0 ? 1.0f : i % 3 == 1 ? 0.0f : 0.5f;
    }
    return mask;
}

std::shared_ptr<Floats> upload(VulkanContext& vc, const std::vector<float>& values) {
    auto buffer = std::make_shared<Floats>(vc, values.size());
    buffer->getBuffer().memcopyFrom(values);
    return buffer;
}

std::vector<float> read(const std::shared_ptr<Floats>& buffer, size_t count) {
    std::vector<float> values(count);
    buffer->getBuffer().memcopyTo(values);
    return values;
}

int64_t readTimestep(const std::shared_ptr<Words>& buffer) {
    std::vector<uint32_t> words(2);
    buffer->getBuffer().memcopyTo(words);
    int64_t value;
    std::memcpy(&value, words.data(), sizeof(value));
    return value;
}

void expectNear(const std::vector<float>& actual, const std::vector<float>& expected, float tolerance) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        ASSERT_NEAR(actual[i], expected[i], tolerance * (1.0f + std::abs(expected[i]))) << "at " << i;
    }
}

// keepOutside() of sampleInpainted().
void keepOutside(std::vector<float>& latents, const std::vector<float>& outside, const std::vector<float>& mask) {
    for (size_t i = 0; i < latents.size(); ++i) {
        const float m = mask[i % kPlane];
        latents[i] = m * latents[i] + (1.0f - m) * outside[i];
    }
}

struct Fixture {
    LcmScheduler scheduler = schedule();
    std::vector<std::vector<float>> noise;
    std::vector<float> mask = mixedMask();
    std::vector<float> encoded = random(kCount, 7, -4.0f, 4.0f);

    Fixture() {
        for (size_t step = 0; step < scheduler.size(); ++step) {
            noise.push_back(random(kCount, 100 + uint32_t(step)));
        }
    }

    std::vector<float> known() const {
        std::vector<float> values = encoded;
        for (auto& value : values) {
            value *= kScale;
        }
        return values;
    }
};

} // namespace

TEST(Sd15GpuInpainting, lcmInitMatchesCpu) {
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    Fixture f;
    auto buffers = LcmInpaintingBuffers::create(vc, kChannels, kPlane, f.noise);
    auto encoded = upload(vc, f.encoded);
    auto latents = std::make_shared<Floats>(vc, kCount);
    auto timestep = std::make_shared<Words>(vc, 2);
    ComputeGraph graph(vc, 1);
    graph.compileFrom(createLcmInit(vc, f.scheduler, kScale, encoded, buffers, latents, timestep));
    buffers.mask->getBuffer(0).memcopyFrom(f.mask);
    graph.submitAndWait(vc.getGraphicsQueue(), 0);

    const auto known = f.known();
    std::vector<float> keptLatent(kCount);
    buffers.known->getBuffer(0).memcopyTo(keptLatent);
    expectNear(keptLatent, known, 1e-6f);
    std::vector<float> expected = f.noise[0];
    keepOutside(expected, f.scheduler.addNoise(0, known, f.noise[0]), f.mask);
    expectNear(read(latents, kCount), expected, 1e-5f);
    EXPECT_EQ(readTimestep(timestep), f.scheduler.getSteps()[0].timestep);
}

TEST(Sd15GpuInpainting, lcmStepMatchesScheduler) {
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    Fixture f;
    const auto known = f.known();
    const std::vector<float> prediction = random(kCount, 3);
    const size_t lastIndex = f.scheduler.size() - 1;
    for (size_t index : {size_t(1), lastIndex}) {
        auto buffers = LcmInpaintingBuffers::create(vc, kChannels, kPlane, f.noise);
        const std::vector<float> start = random(kCount, 11);
        auto latents = upload(vc, start);
        auto predictionBuffer = upload(vc, prediction);
        auto timestep = std::make_shared<Words>(vc, 2);
        timestep->getBuffer().memcopyFrom(std::vector<uint32_t>{12345u, 0u});
        auto decoderInput = upload(vc, std::vector<float>(kCount, -7.0f));
        ComputeGraph graph(vc, 1);
        graph.compileFrom(
            createLcmStep(vc, f.scheduler, index, kScale, latents, predictionBuffer, buffers, timestep, decoderInput));
        buffers.mask->getBuffer(0).memcopyFrom(f.mask);
        buffers.known->getBuffer(0).memcopyFrom(known);
        graph.submitAndWait(vc.getGraphicsQueue(), 0);

        const bool last = index == lastIndex;
        std::vector<float> expected = start, denoised;
        static const std::vector<float> none;
        f.scheduler.step(index, expected, prediction, last ? none : f.noise[index + 1], denoised);
        keepOutside(expected, last ? known : f.scheduler.addNoise(index + 1, known, f.noise[index + 1]), f.mask);
        expectNear(read(latents, kCount), expected, 1e-4f);
        if (last) {
            std::vector<float> scaled = expected;
            for (auto& value : scaled) {
                value /= kScale;
            }
            expectNear(read(decoderInput, kCount), scaled, 1e-4f);
            EXPECT_EQ(readTimestep(timestep), 12345) << "the last step leaves the timestep";
        } else {
            EXPECT_EQ(readTimestep(timestep), f.scheduler.getSteps()[index + 1].timestep);
            expectNear(read(decoderInput, kCount), std::vector<float>(kCount, -7.0f), 0.0f);
        }
    }
}

TEST(Sd15GpuInpainting, chainMatchesSampleInpainted) {
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    Fixture f;
    auto buffers = LcmInpaintingBuffers::create(vc, kChannels, kPlane, f.noise);
    auto encoded = upload(vc, f.encoded);
    auto latents = std::make_shared<Floats>(vc, kCount);
    auto timestep = std::make_shared<Words>(vc, 2);
    auto prediction = std::make_shared<Floats>(vc, kCount);
    auto decoderInput = std::make_shared<Floats>(vc, kCount);
    const std::vector<float> factors = random(kCount, 5, 0.2f, 0.8f);
    auto factorBuffer = upload(vc, factors);

    ComputeGraph init(vc, 1);
    init.compileFrom(createLcmInit(vc, f.scheduler, kScale, encoded, buffers, latents, timestep));
    // The stand-in denoiser: prediction = sample * factor.
    auto predict = std::make_shared<GeneralComputation<>>(vc, "shaders/operator_multiply_scalar_element_wise.comp.spv");
    predict->setInput(latents, 0);
    predict->setInput(factorBuffer, 1);
    predict->setInput(prediction, 2);
    predict->setGroupCountX(kCount);
    ComputeGraph denoise(vc, 1);
    denoise.enableSimultaneousUse();
    denoise.compileFrom(predict);
    std::vector<std::unique_ptr<ComputeGraph>> steps;
    GraphChain chain(vc);
    chain.add(init);
    for (size_t index = 0; index < f.scheduler.size(); ++index) {
        steps.push_back(std::make_unique<ComputeGraph>(vc, 1));
        steps.back()->compileFrom(
            createLcmStep(vc, f.scheduler, index, kScale, latents, prediction, buffers, timestep, decoderInput));
        chain.add(denoise);
        chain.add(*steps.back());
    }
    buffers.mask->getBuffer(0).memcopyFrom(f.mask);
    chain.submitAndWait(vc.getGraphicsQueue());

    const LatentInpainting inpainting{f.mask, f.known()};
    const auto expected = sampleInpainted(
        f.scheduler, f.noise, inpainting,
        [&](const std::vector<float>& sample, int64_t) {
            std::vector<float> result(sample.size());
            for (size_t i = 0; i < sample.size(); ++i) {
                result[i] = sample[i] * factors[i];
            }
            return result;
        },
        kChannels);
    expectNear(read(latents, kCount), expected, 1e-4f);
    std::vector<float> scaled = expected;
    for (auto& value : scaled) {
        value /= kScale;
    }
    expectNear(read(decoderInput, kCount), scaled, 1e-4f);
}
