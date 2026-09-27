#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/tensorelement.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/onnx/onnx_network.hpp"

namespace {

// The checked-in fixed-shape smoke models use 64x64 images so the current
// graph allocator (which retains all intermediates) also works on 8 GB GPUs.
constexpr uint32_t kImageSize = 64;

std::vector<float> readFloats(const std::filesystem::path& path, size_t count) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("Could not open " + path.string());
    if (static_cast<size_t>(input.tellg()) != count * sizeof(float)) {
        throw std::runtime_error("Unexpected tensor size in " + path.string());
    }
    input.seekg(0);
    std::vector<float> values(count);
    input.read(reinterpret_cast<char*>(values.data()), values.size() * sizeof(float));
    if (!input) throw std::runtime_error("Could not read " + path.string());
    return values;
}

void writePpm(const std::filesystem::path& path, const std::vector<float>& nchw) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary);
    if (!output) throw std::runtime_error("Could not create " + path.string());
    output << "P6\n" << kImageSize << " " << kImageSize << "\n255\n";
    const size_t plane = kImageSize * kImageSize;
    for (size_t pixel = 0; pixel < plane; ++pixel) {
        for (size_t channel = 0; channel < 3; ++channel) {
            const float normalized = std::clamp(nchw[channel * plane + pixel], -1.0f, 1.0f);
            const auto byte = static_cast<unsigned char>((normalized + 1.0f) * 127.5f + 0.5f);
            output.write(reinterpret_cast<const char*>(&byte), 1);
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const bool loadOnly = argc > 1 && std::string(argv[1]) == "--load-only";
        const std::filesystem::path modelDirectory = "./data/onnx/sd15";
        const auto encoderPath = modelDirectory / "sd15_vae_encoder.onnx";
        const auto decoderPath = modelDirectory / "sd15_vae_decoder.onnx";
        const auto inputPath = modelDirectory / "lantern_input_f32.bin";
        for (const auto& path : {encoderPath, decoderPath}) {
            if (!std::filesystem::exists(path)) throw std::runtime_error("Missing " + path.string());
        }

        klartraum::HeadlessFrontend frontend;
        auto& context = frontend.getKlartraumEngine().getVulkanContext();
        auto encoder = context.create<klartraum::OnnxNetwork>(encoderPath.string());
        auto decoder = context.create<klartraum::OnnxNetwork>(decoderPath.string());
        decoder->setInputTensor("input", encoder, 0);
        if (loadOnly) {
            std::cout << "Loaded and connected the SD 1.5 VAE encoder and decoder." << std::endl;
            return 0;
        }

        auto image = context.create<klartraum::TensorElement<float>>(
            std::vector<uint32_t>{1, 3, kImageSize, kImageSize});
        encoder->setInputTensor("input", image);
        klartraum::ComputeGraph graph(context, 1);
        graph.compileFrom(decoder);
        image->setData(0, readFloats(inputPath, 3 * kImageSize * kImageSize));
        graph.submitAndWait(context.getGraphicsQueue(), 0);

        auto decoded = std::dynamic_pointer_cast<klartraum::TensorElement<float>>(
            decoder->getOutputElement("output"));
        std::vector<float> values(decoded->getDataElementCount());
        decoded->getDataBuffer(0).memcopyTo(values);
        const auto outputPath = std::filesystem::path("build/TestingOutput/sd15_vae_klartraum.ppm");
        writePpm(outputPath, values);
        std::cout << "Wrote " << outputPath << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "SD 1.5 VAE example failed: " << error.what() << std::endl;
        return 1;
    }
}
