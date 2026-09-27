// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - The simple encoder executes and matches every frozen intermediate.
 * - The simple decoder executes and matches every frozen intermediate.
 * - The Stable Diffusion 1.5 encoder graph loads with the expected output shape.
 * - The Stable Diffusion 1.5 decoder graph loads with the expected output shape.
 * - The Stable Diffusion 1.5 VAE encodes and decodes lantern.jpg within the ONNX reference tolerance.
 * - The Stable Diffusion 1.5 graphs reuse transient tensor storage.
 * - Eligible ONNX Reshape tensors are zero-copy views of their inputs.
 * - The generated fixed-size SD1.5 UNet graph loads with complete operator coverage.
 * - The generated fixed-size SD1.5 UNet executes one denoising prediction against ONNX Runtime.
 **/

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "klartraum/headless_frontend.hpp"
#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/onnx/onnx_network.hpp"
#include "klartraum/computegraph/tensorelement.hpp"
#include "onnx.pb.h"

using namespace klartraum;

namespace {

std::vector<float> readFloatTensor(const std::filesystem::path& path, size_t count) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input || static_cast<size_t>(input.tellg()) != count * sizeof(float)) {
        throw std::runtime_error("Invalid float tensor fixture: " + path.string());
    }
    input.seekg(0);
    std::vector<float> values(count);
    input.read(reinterpret_cast<char*>(values.data()), values.size() * sizeof(float));
    if (!input)
        throw std::runtime_error("Could not read float tensor fixture: " + path.string());
    return values;
}

std::vector<int64_t> readInt64Tensor(const std::filesystem::path& path, size_t count) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input || static_cast<size_t>(input.tellg()) != count * sizeof(int64_t)) {
        throw std::runtime_error("Invalid int64 tensor fixture: " + path.string());
    }
    input.seekg(0);
    std::vector<int64_t> values(count);
    input.read(reinterpret_cast<char*>(values.data()), values.size() * sizeof(int64_t));
    if (!input)
        throw std::runtime_error("Could not read int64 tensor fixture: " + path.string());
    return values;
}

} // namespace

void testLayer(std::shared_ptr<OnnxNetwork> onnxNetwork, std::string layerName) {
    auto element = onnxNetwork->getOutputElement(layerName);
    ASSERT_NE(element, nullptr);
    auto tensor = std::dynamic_pointer_cast<TensorElement<float>>(element);
    ASSERT_NE(tensor, nullptr);

    std::vector<float> dataGPU(tensor->getDataElementCount());
    tensor->getDataBuffer(0).memcopyTo(dataGPU);

    // now compare the output tensor to expected values
    std::vector<float> dataGT = onnxNetwork->getFloatInitializerData(layerName);

    ASSERT_EQ(dataGT.size(), dataGPU.size());

    for (size_t i = 0; i < dataGT.size(); i++) {
        ASSERT_NEAR(dataGT[i], dataGPU[i], 1e-3);
    }
}

// Test execute functionality
TEST(OnnxNetworkTest, ExecuteWithValidEncoderModel) {
    HeadlessFrontend frontend;
    auto& core = frontend.getKlartraumEngine();
    auto& vulkanContext = core.getVulkanContext();

    /*
    STEP 1: create the ONNX network
    */
    std::string modelPath = "./data/onnx/simple_encoder_with_onnx_frozen_intermediates.onnx";

    auto onnxNetwork = vulkanContext.create<OnnxNetwork>(modelPath);

    for (const auto& layer : {"/conv1/Conv_output_0", "/relu/Relu_output_0", "/conv2/Conv_output_0",
                              "/relu_1/Relu_output_0", "/conv3/Conv_output_0", "output"}) {
        onnxNetwork->retainTensor(layer);
    }

    /*
    STEP 2: use the render engine to execute the computegraph so it can be debugged with renderdoc
    */
    core.add(core.createRenderPass());
    core.add(onnxNetwork);
    core.step();

    testLayer(onnxNetwork, "/conv1/Conv_output_0");
    testLayer(onnxNetwork, "/relu/Relu_output_0");
    testLayer(onnxNetwork, "/conv2/Conv_output_0");
    testLayer(onnxNetwork, "/relu_1/Relu_output_0");
    testLayer(onnxNetwork, "/conv3/Conv_output_0");
    testLayer(onnxNetwork, "output");

    return;
}

TEST(OnnxNetworkTest, ExecuteWithValidDecoderModel) {
    HeadlessFrontend frontend;
    auto& core = frontend.getKlartraumEngine();
    auto& vulkanContext = core.getVulkanContext();

    /*
    STEP 1: create the ONNX network
    */
    std::string modelPath = "./data/onnx/simple_decoder_with_onnx_frozen_intermediates.onnx";

    auto onnxNetwork = vulkanContext.create<OnnxNetwork>(modelPath);

    for (const auto& layer :
         {"/deconv1/ConvTranspose_output_0", "/relu/Relu_output_0", "/deconv2/ConvTranspose_output_0",
          "/relu_1/Relu_output_0", "/deconv3/ConvTranspose_output_0", "output"}) {
        onnxNetwork->retainTensor(layer);
    }

    /*
    STEP 2: use the render engine to execute the computegraph so it can be debugged with renderdoc
    */
    core.add(core.createRenderPass());
    core.add(onnxNetwork);
    core.step();

    testLayer(onnxNetwork, "/deconv1/ConvTranspose_output_0");
    testLayer(onnxNetwork, "/relu/Relu_output_0");
    testLayer(onnxNetwork, "/deconv2/ConvTranspose_output_0");
    testLayer(onnxNetwork, "/relu_1/Relu_output_0");
    testLayer(onnxNetwork, "/deconv3/ConvTranspose_output_0");
    testLayer(onnxNetwork, "output");

    return;
}

TEST(OnnxNetworkTest, LoadsStableDiffusion15Encoder) {
    HeadlessFrontend frontend;
    auto& context = frontend.getKlartraumEngine().getVulkanContext();
    auto network = context.create<OnnxNetwork>("./data/onnx/sd15/sd15_vae_encoder.onnx");
    auto output = std::dynamic_pointer_cast<TensorElement<float>>(network->getOutputElement("output"));
    ASSERT_NE(output, nullptr);
    EXPECT_EQ(output->getDimensions(), (std::vector<uint32_t>{1, 4, 8, 8}));
}

TEST(OnnxNetworkTest, LoadsStableDiffusion15Decoder) {
    HeadlessFrontend frontend;
    auto& context = frontend.getKlartraumEngine().getVulkanContext();
    auto network = context.create<OnnxNetwork>("./data/onnx/sd15/sd15_vae_decoder.onnx");
    auto output = std::dynamic_pointer_cast<TensorElement<float>>(network->getOutputElement("output"));
    ASSERT_NE(output, nullptr);
    EXPECT_EQ(output->getDimensions(), (std::vector<uint32_t>{1, 3, 64, 64}));
}

TEST(OnnxNetworkTest, StableDiffusion15UsesTransientStoragePlan) {
    HeadlessFrontend frontend;
    auto& context = frontend.getKlartraumEngine().getVulkanContext();
    auto network = context.create<OnnxNetwork>("./data/onnx/sd15/sd15_vae_decoder.onnx");
    const auto& stats = network->getMemoryPlanStats();
    EXPECT_GT(stats.tensorCount, stats.slotCount);
    EXPECT_GT(stats.logicalBytes, stats.allocatedBytes);
    EXPECT_GE(stats.allocatedBytes, stats.peakLiveBytes);
    EXPECT_GT(stats.viewAliasCount, 0);

    auto reshapeInput = std::dynamic_pointer_cast<TensorElementInterface>(
        network->getOutputElement("/decoder/up_blocks.3/resnets.2/Add_output_0"));
    auto reshapeOutput = std::dynamic_pointer_cast<TensorElementInterface>(
        network->getOutputElement("/decoder/conv_norm_out/Reshape_output_0"));
    ASSERT_NE(reshapeInput, nullptr);
    ASSERT_NE(reshapeOutput, nullptr);
    EXPECT_EQ(reshapeInput->getStorageIdentity(), reshapeOutput->getStorageIdentity());
}

TEST(OnnxNetworkTest, LoadsStableDiffusion15Denoiser) {
    const std::filesystem::path modelPath = "./data/onnx/sd15_denoiser_128/sd15_unet.onnx";
    if (!std::filesystem::exists(modelPath))
        GTEST_SKIP() << "Generate the denoiser with scripts/sd15_onnx/export_denoiser.py";
    HeadlessFrontend frontend;
    auto& context = frontend.getKlartraumEngine().getVulkanContext();
    auto network = context.create<OnnxNetwork>(modelPath.string());
    auto output = std::dynamic_pointer_cast<TensorElement<float>>(network->getOutputElement("noise_prediction"));
    ASSERT_NE(output, nullptr);
    EXPECT_EQ(output->getDimensions(), (std::vector<uint32_t>{2, 4, 16, 16}));
    const auto& stats = network->getMemoryPlanStats();
    EXPECT_GT(stats.tensorCount, stats.slotCount);
    EXPECT_GT(stats.logicalBytes, stats.allocatedBytes);
    EXPECT_GT(stats.viewAliasCount, 0);
}

TEST(OnnxNetworkTest, ExecutesStableDiffusion15DenoiserStep) {
    constexpr size_t sampleElements = 2 * 4 * 16 * 16;
    constexpr size_t embeddingElements = 2 * 77 * 768;
    const std::filesystem::path modelDirectory = "./data/onnx/sd15_denoiser_128";
    const auto modelPath = modelDirectory / "sd15_unet.onnx";
    if (!std::filesystem::exists(modelPath))
        GTEST_SKIP() << "Generate the denoiser with scripts/sd15_onnx/export_denoiser.py";

    HeadlessFrontend frontend;
    auto& context = frontend.getKlartraumEngine().getVulkanContext();
    auto network = context.create<OnnxNetwork>(modelPath.string());
    auto sample = context.create<TensorElement<float>>(std::vector<uint32_t>{2, 4, 16, 16});
    auto timestep = context.create<TensorElement<int64_t>>(std::vector<uint32_t>{1});
    auto embeddings = context.create<TensorElement<float>>(std::vector<uint32_t>{2, 77, 768});
    network->setInputTensor("sample", sample);
    network->setInputTensor("timestep", timestep);
    network->setInputTensor("encoder_hidden_states", embeddings);

    ComputeGraph graph(context, 1);
    graph.compileFrom(network);
    sample->setData(0, readFloatTensor(modelDirectory / "unet_sample_f32.bin", sampleElements));
    timestep->setData(0, readInt64Tensor(modelDirectory / "unet_timestep_i64.bin", 1));
    embeddings->setData(0, readFloatTensor(modelDirectory / "prompt_embeddings_f32.bin", embeddingElements));
    graph.submitAndWait(context.getGraphicsQueue(), 0);

    auto output = std::dynamic_pointer_cast<TensorElement<float>>(network->getOutputElement("noise_prediction"));
    ASSERT_NE(output, nullptr);
    std::vector<float> actual(output->getDataElementCount());
    output->getDataBuffer(0).memcopyTo(actual);
    const auto expected = readFloatTensor(modelDirectory / "unet_reference_f32.bin", sampleElements);
    ASSERT_EQ(actual.size(), expected.size());
    float maximumAbsoluteError = 0.0f;
    for (size_t index = 0; index < actual.size(); ++index) {
        maximumAbsoluteError = std::max(maximumAbsoluteError, std::abs(actual[index] - expected[index]));
    }
    EXPECT_LE(maximumAbsoluteError, 2e-2f);
}

TEST(OnnxNetworkTest, ExecutesStableDiffusion15VaeOnLantern) {
    constexpr size_t imageElementCount = 3 * 64 * 64;
    const std::filesystem::path modelDirectory = "./data/onnx/sd15";

    HeadlessFrontend frontend;
    auto& context = frontend.getKlartraumEngine().getVulkanContext();
    auto encoder = context.create<OnnxNetwork>((modelDirectory / "sd15_vae_encoder.onnx").string());
    auto decoder = context.create<OnnxNetwork>((modelDirectory / "sd15_vae_decoder.onnx").string());
    auto image = context.create<TensorElement<float>>(std::vector<uint32_t>{1, 3, 64, 64});
    encoder->setInputTensor("input", image);
    decoder->setInputTensor("input", encoder, 0);

    ComputeGraph graph(context, 1);
    graph.compileFrom(decoder);
    image->setData(0, readFloatTensor(modelDirectory / "lantern_input_f32.bin", imageElementCount));
    graph.submitAndWait(context.getGraphicsQueue(), 0);

    auto output = std::dynamic_pointer_cast<TensorElement<float>>(decoder->getOutputElement("output"));
    ASSERT_NE(output, nullptr);
    std::vector<float> actual(output->getDataElementCount());
    output->getDataBuffer(0).memcopyTo(actual);
    const auto expected = readFloatTensor(modelDirectory / "lantern_reference_f32.bin", imageElementCount);
    ASSERT_EQ(actual.size(), expected.size());

    float maximumAbsoluteError = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) {
        maximumAbsoluteError = std::max(maximumAbsoluteError, std::abs(actual[i] - expected[i]));
    }
    EXPECT_LE(maximumAbsoluteError, 2e-3f);
}
