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
 * - imageToTensorMatchesCpu: an image read into a tensor in [-1, 1] equals bgraToTensor()
 * - latentMaskMatchesCpu: the latent mask of a margin mask and a body mask equals closeUpLatentMask()
 * - windowMatchesCpu: over several frames of moving joints, the window state and the scene and skeleton cameras
 *   written to their uniform buffers equal CloseUpWindow::update(), sceneCamera() and skeletonCamera()
 * - pasteMatchesCpu: a generated close-up pasted into a scene equals pasteGenerated() within one grey level
 * - compositeMatchesCpu: the pasted close-up composited into a wide view equals compositeCloseUp() within one grey
 *   level
 **/

#include <cstring>
#include <glm/gtc/matrix_transform.hpp>
#include <memory>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/graphchain.hpp"
#include "klartraum/computegraph/noop.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/offscreen_target.hpp"
#include "klartraum/sd15/closeup_inpainting.hpp"
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

namespace {

constexpr VkImageLayout kRendered = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

// An OffscreenTarget holding `bgra`, left in TRANSFER_SRC_OPTIMAL like a rendered one.
std::shared_ptr<OffscreenTarget> imageOf(VulkanContext& vc, const std::vector<uint8_t>& bgra, VkExtent2D extent) {
    auto target = std::make_shared<OffscreenTarget>(vc, extent, 1u);
    target->setClear(false);
    VulkanBuffer<uint8_t> staging(vc, uint32_t(bgra.size()));
    staging.memcopyFrom(bgra);
    vc.submitImmediate([&](VkCommandBuffer commandBuffer) {
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = target->getImage(0);
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                             nullptr, 0, nullptr, 1, &barrier);
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {extent.width, extent.height, 1};
        vkCmdCopyBufferToImage(commandBuffer, staging.getBuffer(), target->getImage(0),
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = kRendered;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0,
                             nullptr, 0, nullptr, 1, &barrier);
    });
    return target;
}

std::vector<uint8_t> randomImage(size_t pixels, uint32_t seed) {
    std::mt19937 generator(seed);
    std::uniform_int_distribution<int> byte(0, 255);
    std::vector<uint8_t> bgra(4 * pixels);
    for (auto& value : bgra) {
        value = uint8_t(byte(generator));
    }
    return bgra;
}

// A mask image: red set inside the ellipse and, with `scattered`, faint red 1 at a few other pixels.
std::vector<uint8_t> ellipseMask(int width, int height, glm::vec2 centre, glm::vec2 radius, uint8_t red = 255,
                                 bool scattered = false) {
    std::vector<uint8_t> bgra(4 * size_t(width) * size_t(height), 0);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const glm::vec2 d = (glm::vec2(x, y) - centre) / radius;
            const size_t i = size_t(y) * size_t(width) + size_t(x);
            if (glm::dot(d, d) <= 1.0f) {
                bgra[4 * i + 2] = red;
            } else if (scattered && (x * 7 + y * 13) % 97 == 0) {
                bgra[4 * i + 2] = 1;
            }
            bgra[4 * i + 3] = 255;
        }
    }
    return bgra;
}

std::vector<uint8_t> unpack(const std::vector<uint32_t>& packed) {
    std::vector<uint8_t> bytes(4 * packed.size());
    std::memcpy(bytes.data(), packed.data(), bytes.size());
    return bytes;
}

void expectBytesNear(const std::vector<uint8_t>& actual, const std::vector<uint8_t>& expected, int tolerance,
                     size_t stride = 1, size_t channels = 1) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); i += stride) {
        for (size_t c = 0; c < channels; ++c) {
            ASSERT_LE(std::abs(int(actual[i + c]) - int(expected[i + c])), tolerance) << "at byte " << i + c;
        }
    }
}

void expectMatrixNear(const glm::mat4& actual, const glm::mat4& expected) {
    for (int c = 0; c < 4; ++c) {
        for (int r = 0; r < 4; ++r) {
            EXPECT_NEAR(actual[c][r], expected[c][r], 1e-4f * (1.0f + std::abs(expected[c][r])));
        }
    }
}

} // namespace

TEST(Sd15GpuInpainting, imageToTensorMatchesCpu) {
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    const VkExtent2D extent{48, 32};
    const size_t pixels = size_t(extent.width) * extent.height;
    const auto bgra = randomImage(pixels, 21);
    auto image = imageOf(vc, bgra, extent);
    auto tensor = std::make_shared<Floats>(vc, 3 * pixels);
    ComputeGraph graph(vc, 1);
    graph.compileFrom(createImageToTensor(vc, image, kRendered, tensor, extent, -1.0f, 1.0f));
    graph.submitAndWait(vc.getGraphicsQueue(), 0);
    expectNear(read(tensor, 3 * pixels), bgraToTensor(bgra, pixels, -1.0f, 1.0f), 1e-6f);
}

TEST(Sd15GpuInpainting, latentMaskMatchesCpu) {
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    const int size = 128;
    const VkExtent2D extent{uint32_t(size), uint32_t(size)};
    const auto margin = ellipseMask(size, size, {60.0f, 70.0f}, {40.0f, 30.0f}, 255, true);
    const auto body = ellipseMask(size, size, {58.0f, 72.0f}, {20.0f, 12.0f}, 3);
    auto marginImage = imageOf(vc, margin, extent);
    auto bodyImage = imageOf(vc, body, extent);
    const uint32_t latentPixels = (size / 8) * (size / 8);
    auto latentMaskBuffer = std::make_shared<Floats>(vc, latentPixels);
    ComputeGraph graph(vc, 1);
    graph.compileFrom(
        createCloseUpLatentMask(vc, marginImage, kRendered, bodyImage, kRendered, latentMaskBuffer, extent));
    graph.submitAndWait(vc.getGraphicsQueue(), 0);
    const auto expected = closeUpLatentMask(redChannel(margin), redChannel(body), size, size);
    const auto actual = read(latentMaskBuffer, latentPixels);
    EXPECT_EQ(actual, expected);
    EXPECT_NE(std::count(expected.begin(), expected.end(), 0.5f), 0) << "the test has a margin ring";
    EXPECT_NE(std::count(expected.begin(), expected.end(), 1.0f), 0) << "the test has a body";
}

TEST(Sd15GpuInpainting, windowMatchesCpu) {
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    const float aspect = 16.0f / 9.0f;
    CameraMVP full{};
    full.model = glm::mat4(1.0f);
    full.view = glm::lookAt(glm::vec3(0.5f, 0.6f, 1.4f), glm::vec3(0.0f, 0.2f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    full.proj = glm::perspective(glm::radians(45.0f), aspect, 0.01f, 10.0f);
    full.proj[1][1] *= -1.0f;
    full.cameraWorldPos = glm::vec4(0.5f, 0.6f, 1.4f, 1.0f);
    const glm::mat4 characterToScene =
        glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(0.1f, 0.0f, -0.2f)), glm::vec3(0.2f));
    constexpr uint32_t kJoints = 18;

    auto joints = std::make_shared<BufferElementSinglePath<VulkanBuffer<glm::vec4>>>(vc, kJoints);
    auto fullCamera = std::make_shared<CameraUboType>();
    auto sceneCamera = std::make_shared<CameraUboType>();
    auto skeletonCamera = std::make_shared<CameraUboType>();
    auto window =
        createCloseUpWindow(vc, joints, kJoints, fullCamera, characterToScene, aspect, sceneCamera, skeletonCamera);
    auto cameras = std::make_shared<NoOp>(vc);
    cameras->addDependency(sceneCamera);
    cameras->addDependency(skeletonCamera);
    ComputeGraph graph(vc, 1);
    graph.compileFrom(cameras);
    fullCamera->ubo = full;
    fullCamera->update(0);

    CloseUpWindow expected;
    for (int frame = 0; frame < 4; ++frame) {
        std::vector<glm::vec4> positions(kJoints);
        for (uint32_t j = 0; j < kJoints; ++j) {
            // A body whose limbs swing, so the size changes from frame to frame.
            const float swing = 0.3f * std::sin(float(frame) + float(j));
            positions[j] =
                glm::vec4(0.2f * float(frame) + swing * float(j % 3), 1.7f * float(j) / kJoints, 0.1f * swing, 1.0f);
        }
        joints->getBuffer().memcopyFrom(positions);
        graph.submitAndWait(vc.getGraphicsQueue(), 0);
        expected.update(positions, full, characterToScene, aspect);

        std::vector<float> state(8);
        window.state->getBuffer().memcopyTo(state);
        EXPECT_NEAR(state[0], expected.centre.x, 1e-4f) << "frame " << frame;
        EXPECT_NEAR(state[1], expected.centre.y, 1e-4f) << "frame " << frame;
        EXPECT_NEAR(state[2], expected.halfExtent.x, 1e-4f) << "frame " << frame;
        EXPECT_NEAR(state[3], expected.halfExtent.y, 1e-4f) << "frame " << frame;
        EXPECT_EQ(state[5], 1.0f);
        const CameraMVP scene = sceneCamera->read(0), skeleton = skeletonCamera->read(0);
        expectMatrixNear(scene.proj, expected.sceneCamera(full).proj);
        expectMatrixNear(scene.view, full.view);
        expectMatrixNear(scene.model, full.model);
        expectMatrixNear(skeleton.model, characterToScene);
        expectMatrixNear(skeleton.proj, expected.skeletonCamera(full, characterToScene).proj);
    }
}

TEST(Sd15GpuInpainting, pasteMatchesCpu) {
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    const int size = 64;
    const VkExtent2D extent{uint32_t(size), uint32_t(size)};
    const size_t pixels = size_t(size) * size_t(size);
    const auto scene = randomImage(pixels, 31);
    const auto body = ellipseMask(size, size, {30.0f, 34.0f}, {12.0f, 20.0f});
    std::vector<float> generated = random(3 * pixels, 33, -1.3f, 1.3f);
    auto sceneImage = imageOf(vc, scene, extent);
    auto bodyImage = imageOf(vc, body, extent);
    auto generatedBuffer = upload(vc, generated);
    auto closeUp = std::make_shared<Words>(vc, pixels);
    ComputeGraph graph(vc, 1);
    graph.compileFrom(
        createCloseUpPaste(vc, generatedBuffer, sceneImage, kRendered, bodyImage, kRendered, closeUp, size));
    graph.submitAndWait(vc.getGraphicsQueue(), 0);

    std::vector<uint32_t> packed(pixels);
    closeUp->getBuffer().memcopyTo(packed);
    const auto actual = unpack(packed);
    const auto expected = pasteGenerated(scene, generated, redChannel(body), size);
    expectBytesNear(actual, expected.image, 1, 4, 3);
    for (size_t i = 0; i < pixels; ++i) {
        ASSERT_LE(std::abs(int(actual[4 * i + 3]) - int(expected.alpha[4 * i])), 1) << "alpha at " << i;
    }
}

TEST(Sd15GpuInpainting, compositeMatchesCpu) {
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    const int width = 96, height = 54, size = 32;
    const VkExtent2D extent{uint32_t(width), uint32_t(height)};
    const size_t pixels = size_t(width) * size_t(height), closeUpPixels = size_t(size) * size_t(size);
    const auto scene = randomImage(pixels, 41);
    const auto mask = ellipseMask(width, height, {50.0f, 25.0f}, {10.0f, 14.0f});
    PastedCloseUp pasted{randomImage(closeUpPixels, 43), randomImage(closeUpPixels, 44)};
    CloseUpWindow window;
    window.centre = {0.05f, -0.07f};
    window.halfExtent = {0.3f / (16.0f / 9.0f), 0.3f};

    std::vector<uint32_t> packed(closeUpPixels);
    for (size_t i = 0; i < closeUpPixels; ++i) {
        packed[i] = uint32_t(pasted.image[4 * i]) | uint32_t(pasted.image[4 * i + 1]) << 8 |
                    uint32_t(pasted.image[4 * i + 2]) << 16 | uint32_t(pasted.alpha[4 * i]) << 24;
    }
    auto closeUp = std::make_shared<Words>(vc, closeUpPixels);
    closeUp->getBuffer().memcopyFrom(packed);
    auto state = upload(
        vc, {window.centre.x, window.centre.y, window.halfExtent.x, window.halfExtent.y, 0.3f, 1.0f, 0.0f, 0.0f});
    auto sceneImage = imageOf(vc, scene, extent);
    auto maskImage = imageOf(vc, mask, extent);
    auto frame = std::make_shared<Words>(vc, pixels);
    ComputeGraph graph(vc, 1);
    graph.compileFrom(
        createCloseUpComposite(vc, sceneImage, kRendered, maskImage, kRendered, closeUp, state, frame, extent, size));
    graph.submitAndWait(vc.getGraphicsQueue(), 0);

    std::vector<uint32_t> result(pixels);
    frame->getBuffer().memcopyTo(result);
    const auto expected = compositeCloseUp(scene, mask, pasted, size, width, height, window);
    expectBytesNear(unpack(result), expected, 1);
    EXPECT_NE(expected, scene) << "the test pastes something";
}
