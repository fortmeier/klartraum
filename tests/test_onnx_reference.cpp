/**
 * TESTS:
 * - encoderMatchesReference: the plain sample encoder reproduces the reference output stored in the
 *   frozen encoder for the reference input; the plain model preloads no expected values, so every output
 *   element has to be computed
 * - decoderMatchesReference: the same for the plain sample decoder
 * - encoderDecoderChainMatchesReference: the plain encoder and decoder chained with setInputTensor()
 *   reproduce the decoder's reference output from the encoder's reference input
 **/

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/tensorelement.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/onnx/onnx_network.hpp"

using namespace klartraum;

namespace {

const std::string kEncoder = "./data/onnx/simple_encoder.onnx";
const std::string kDecoder = "./data/onnx/simple_decoder.onnx";
// The frozen models store their reference input, intermediates and output
// as initializers; they only serve as the source of expected values here.
const std::string kFrozenEncoder = "./data/onnx/simple_encoder_with_onnx_frozen_intermediates.onnx";
const std::string kFrozenDecoder = "./data/onnx/simple_decoder_with_onnx_frozen_intermediates.onnx";

float maxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.0f;
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
        m = std::max(m, std::abs(a[i] - b[i]));
    }
    return m;
}

std::vector<float> readTensor(const std::shared_ptr<OnnxNetwork>& network, const std::string& name) {
    auto tensor = std::dynamic_pointer_cast<TensorElement<float>>(network->getOutputElement(name));
    EXPECT_NE(tensor, nullptr) << name;
    if (!tensor) {
        return {};
    }
    std::vector<float> data(tensor->getDataElementCount());
    tensor->getDataBuffer(0).memcopyTo(data);
    return data;
}

// Runs `network` once on `input` in a single-path graph and returns the named outputs.
std::vector<std::vector<float>> runOnce(VulkanContext& vc, const std::shared_ptr<OnnxNetwork>& network,
                                        const std::vector<uint32_t>& inputShape, const std::vector<float>& input,
                                        const std::vector<std::string>& outputs) {
    auto tensor = vc.create<TensorElement<float>>(inputShape);
    network->setInputTensor("input", tensor, -1);
    ComputeGraph graph(vc, 1);
    graph.compileFrom(network);
    tensor->getDataBuffer(0).memcopyFrom(input);
    graph.submitAndWait(vc.getGraphicsQueue(), 0);
    std::vector<std::vector<float>> result;
    for (const auto& name : outputs) {
        result.push_back(readTensor(network, name));
    }
    return result;
}

bool samplesExist() {
    for (const auto& path : {kEncoder, kDecoder, kFrozenEncoder, kFrozenDecoder}) {
        if (!std::filesystem::exists(path)) {
            return false;
        }
    }
    return true;
}

} // namespace

TEST(OnnxReferenceTest, encoderMatchesReference) {
    if (!samplesExist()) {
        GTEST_SKIP() << "ONNX samples not found";
    }
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    auto frozen = vc.create<OnnxNetwork>(kFrozenEncoder);
    auto encoder = vc.create<OnnxNetwork>(kEncoder);

    const std::vector<std::string> layers{"/conv1/Conv_output_0", "/relu/Relu_output_0", "/conv2/Conv_output_0",
                                          "/relu_1/Relu_output_0", "/conv3/Conv_output_0", "output"};
    const auto actual = runOnce(vc, encoder, {1, 3, 128, 128}, frozen->getFloatInitializerData("input"), layers);
    for (size_t i = 0; i < layers.size(); ++i) {
        const auto expected = frozen->getFloatInitializerData(layers[i]);
        ASSERT_EQ(actual[i].size(), expected.size()) << layers[i];
        EXPECT_LT(maxAbsDiff(actual[i], expected), 1e-3f) << layers[i];
    }
}

TEST(OnnxReferenceTest, decoderMatchesReference) {
    if (!samplesExist()) {
        GTEST_SKIP() << "ONNX samples not found";
    }
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    auto frozen = vc.create<OnnxNetwork>(kFrozenDecoder);
    auto decoder = vc.create<OnnxNetwork>(kDecoder);

    const std::vector<std::string> layers{"/deconv1/ConvTranspose_output_0", "/relu/Relu_output_0",
                                          "/deconv2/ConvTranspose_output_0", "/relu_1/Relu_output_0",
                                          "/deconv3/ConvTranspose_output_0", "output"};
    const auto actual = runOnce(vc, decoder, {1, 128, 16, 16}, frozen->getFloatInitializerData("input"), layers);
    for (size_t i = 0; i < layers.size(); ++i) {
        const auto expected = frozen->getFloatInitializerData(layers[i]);
        ASSERT_EQ(actual[i].size(), expected.size()) << layers[i];
        EXPECT_LT(maxAbsDiff(actual[i], expected), 1e-3f) << layers[i];
    }
}

TEST(OnnxReferenceTest, encoderDecoderChainMatchesReference) {
    if (!samplesExist()) {
        GTEST_SKIP() << "ONNX samples not found";
    }
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    auto frozenEncoder = vc.create<OnnxNetwork>(kFrozenEncoder);
    auto frozenDecoder = vc.create<OnnxNetwork>(kFrozenDecoder);
    // The references form one chain: the decoder's input is the encoder's output.
    ASSERT_LT(maxAbsDiff(frozenDecoder->getFloatInitializerData("input"),
                         frozenEncoder->getFloatInitializerData("output")), 1e-5f);

    auto encoder = vc.create<OnnxNetwork>(kEncoder);
    auto decoder = vc.create<OnnxNetwork>(kDecoder);
    auto tensor = vc.create<TensorElement<float>>(std::vector<uint32_t>{1, 3, 128, 128});
    encoder->setInputTensor("input", tensor, -1);
    decoder->setInputTensor("input", encoder, 0);

    ComputeGraph graph(vc, 1);
    graph.compileFrom(decoder);
    tensor->getDataBuffer(0).memcopyFrom(frozenEncoder->getFloatInitializerData("input"));
    graph.submitAndWait(vc.getGraphicsQueue(), 0);

    EXPECT_LT(maxAbsDiff(readTensor(decoder, "output"), frozenDecoder->getFloatInitializerData("output")), 1e-3f);
}
