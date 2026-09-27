// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cmath>
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

std::vector<float> readFloats(const std::filesystem::path& path, size_t count) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input)
        throw std::runtime_error("Could not open " + path.string());
    if (static_cast<size_t>(input.tellg()) != count * sizeof(float)) {
        throw std::runtime_error("Unexpected tensor size in " + path.string());
    }
    input.seekg(0);
    std::vector<float> values(count);
    input.read(reinterpret_cast<char*>(values.data()), values.size() * sizeof(float));
    if (!input)
        throw std::runtime_error("Could not read " + path.string());
    return values;
}

void writePpm(const std::filesystem::path& path, const std::vector<float>& nchw, uint32_t imageSize) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary);
    if (!output)
        throw std::runtime_error("Could not create " + path.string());
    output << "P6\n" << imageSize << " " << imageSize << "\n255\n";
    const size_t plane = static_cast<size_t>(imageSize) * imageSize;
    for (size_t pixel = 0; pixel < plane; ++pixel) {
        for (size_t channel = 0; channel < 3; ++channel) {
            const float normalized = std::clamp(nchw[channel * plane + pixel], -1.0f, 1.0f);
            const auto byte = static_cast<unsigned char>((normalized + 1.0f) * 127.5f + 0.5f);
            output.write(reinterpret_cast<const char*>(&byte), 1);
        }
    }
}

struct Options {
    bool loadOnly = false;
    uint32_t imageSize = 64;
    std::filesystem::path modelDirectory = "./data/onnx/sd15";
    std::filesystem::path outputPath = "build/TestingOutput/sd15_vae_klartraum.ppm";
};

Options parseOptions(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--load-only") {
            options.loadOnly = true;
        } else if (argument == "--size" && index + 1 < argc) {
            options.imageSize = static_cast<uint32_t>(std::stoul(argv[++index]));
        } else if (argument == "--model-dir" && index + 1 < argc) {
            options.modelDirectory = argv[++index];
        } else if (argument == "--output" && index + 1 < argc) {
            options.outputPath = argv[++index];
        } else {
            throw std::runtime_error("Unknown or incomplete argument: " + argument);
        }
    }
    if (options.imageSize == 0 || options.imageSize % 8 != 0) {
        throw std::runtime_error("--size must be a positive multiple of 8");
    }
    return options;
}

void printMemoryPlan(const char* name, const klartraum::OnnxMemoryPlanStats& stats) {
    constexpr double bytesPerMiB = 1024.0 * 1024.0;
    std::cout << name << " transient storage: " << stats.logicalBytes / bytesPerMiB << " logical MiB -> "
              << stats.allocatedBytes / bytesPerMiB << " allocated MiB in " << stats.slotCount << " slots ("
              << stats.viewAliasCount << " zero-copy views)" << std::endl;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parseOptions(argc, argv);
        const auto& modelDirectory = options.modelDirectory;
        const auto encoderPath = modelDirectory / "sd15_vae_encoder.onnx";
        const auto decoderPath = modelDirectory / "sd15_vae_decoder.onnx";
        const auto inputPath = modelDirectory / "lantern_input_f32.bin";
        for (const auto& path : {encoderPath, decoderPath}) {
            if (!std::filesystem::exists(path))
                throw std::runtime_error("Missing " + path.string());
        }

        klartraum::HeadlessFrontend frontend;
        auto& context = frontend.getKlartraumEngine().getVulkanContext();
        if (options.loadOnly) {
            auto encoder = context.create<klartraum::OnnxNetwork>(encoderPath.string());
            auto decoder = context.create<klartraum::OnnxNetwork>(decoderPath.string());
            printMemoryPlan("Encoder", encoder->getMemoryPlanStats());
            printMemoryPlan("Decoder", decoder->getMemoryPlanStats());
            std::cout << "Loaded the SD 1.5 VAE encoder and decoder." << std::endl;
            return 0;
        }

        const size_t imageElements = 3ull * options.imageSize * options.imageSize;
        std::vector<float> latentValues;
        {
            auto encoder = context.create<klartraum::OnnxNetwork>(encoderPath.string());
            printMemoryPlan("Encoder", encoder->getMemoryPlanStats());
            auto image = context.create<klartraum::TensorElement<float>>(
                std::vector<uint32_t>{1, 3, options.imageSize, options.imageSize});
            encoder->setInputTensor("input", image);
            klartraum::ComputeGraph graph(context, 1);
            graph.compileFrom(encoder);
            image->setData(0, readFloats(inputPath, imageElements));
            graph.submitAndWait(context.getGraphicsQueue(), 0);
            auto latent =
                std::dynamic_pointer_cast<klartraum::TensorElement<float>>(encoder->getOutputElement("output"));
            latentValues.resize(latent->getDataElementCount());
            latent->getDataBuffer(0).memcopyTo(latentValues);
        }

        std::vector<float> values;
        {
            auto decoder = context.create<klartraum::OnnxNetwork>(decoderPath.string());
            printMemoryPlan("Decoder", decoder->getMemoryPlanStats());
            const uint32_t latentSize = options.imageSize / 8;
            auto latent =
                context.create<klartraum::TensorElement<float>>(std::vector<uint32_t>{1, 4, latentSize, latentSize});
            decoder->setInputTensor("input", latent);
            klartraum::ComputeGraph graph(context, 1);
            graph.compileFrom(decoder);
            latent->setData(0, latentValues);
            graph.submitAndWait(context.getGraphicsQueue(), 0);
            auto decoded =
                std::dynamic_pointer_cast<klartraum::TensorElement<float>>(decoder->getOutputElement("output"));
            values.resize(decoded->getDataElementCount());
            decoded->getDataBuffer(0).memcopyTo(values);
        }

        const auto referencePath = modelDirectory / "lantern_reference_f32.bin";
        if (std::filesystem::exists(referencePath)) {
            const auto reference = readFloats(referencePath, imageElements);
            float maximumAbsoluteError = 0.0f;
            double absoluteErrorSum = 0.0;
            double squaredErrorSum = 0.0;
            for (size_t index = 0; index < values.size(); ++index) {
                const float error = std::abs(values[index] - reference[index]);
                maximumAbsoluteError = std::max(maximumAbsoluteError, error);
                absoluteErrorSum += error;
                squaredErrorSum += static_cast<double>(error) * error;
            }
            const double meanAbsoluteError = absoluteErrorSum / values.size();
            const double rootMeanSquaredError = std::sqrt(squaredErrorSum / values.size());
            std::cout << "Error versus ONNX Runtime: max=" << maximumAbsoluteError << ", mean=" << meanAbsoluteError
                      << ", rmse=" << rootMeanSquaredError << std::endl;
            // The 64x64 smoke graph remains tightly bounded. At 512x512 the
            // attention reductions are much longer and legitimate Vulkan/CPU
            // floating-point ordering differences accumulate by a few pixels.
            const float maximumTolerance = options.imageSize >= 512 ? 3e-2f : 2e-3f;
            if (maximumAbsoluteError > maximumTolerance) {
                throw std::runtime_error("Klartraum output does not match the ONNX reference");
            }
        }
        writePpm(options.outputPath, values, options.imageSize);
        std::cout << "Wrote " << options.outputPath << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "SD 1.5 VAE example failed: " << error.what() << std::endl;
        return 1;
    }
}
