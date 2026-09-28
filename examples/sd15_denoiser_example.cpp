// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/tensorelement.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/onnx/onnx_network.hpp"

namespace {

template <typename T>
std::vector<T> readTensor(const std::filesystem::path& path, size_t count) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input || static_cast<size_t>(input.tellg()) != count * sizeof(T)) {
        throw std::runtime_error("Unexpected tensor size in " + path.string());
    }
    input.seekg(0);
    std::vector<T> values(count);
    input.read(reinterpret_cast<char*>(values.data()), values.size() * sizeof(T));
    if (!input)
        throw std::runtime_error("Could not read " + path.string());
    return values;
}

template <typename T>
std::vector<T> readTensor(const std::filesystem::path& path) {
    const auto bytes = std::filesystem::file_size(path);
    if (bytes % sizeof(T) != 0) {
        throw std::runtime_error("Invalid tensor byte count in " + path.string());
    }
    return readTensor<T>(path, static_cast<size_t>(bytes / sizeof(T)));
}

float maximumError(const std::vector<float>& actual, const std::vector<float>& expected) {
    if (actual.size() != expected.size())
        throw std::runtime_error("Reference tensor size mismatch");
    float result = 0.0f;
    for (size_t index = 0; index < actual.size(); ++index) {
        if (!std::isfinite(actual[index]) || !std::isfinite(expected[index])) {
            return std::numeric_limits<float>::infinity();
        }
        result = std::max(result, std::abs(actual[index] - expected[index]));
    }
    return result;
}

float maximumMagnitude(const std::vector<float>& values) {
    float result = 0.0f;
    for (float value : values) {
        if (!std::isfinite(value))
            return std::numeric_limits<float>::infinity();
        result = std::max(result, std::abs(value));
    }
    return result;
}

void writePpm(const std::filesystem::path& path, const std::vector<float>& nchw, uint32_t size) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary);
    if (!output)
        throw std::runtime_error("Could not create " + path.string());
    output << "P6\n" << size << " " << size << "\n255\n";
    const size_t plane = static_cast<size_t>(size) * size;
    for (size_t pixel = 0; pixel < plane; ++pixel) {
        for (size_t channel = 0; channel < 3; ++channel) {
            const float normalized = std::clamp(nchw[channel * plane + pixel], -1.0f, 1.0f);
            const auto byte = static_cast<unsigned char>((normalized + 1.0f) * 127.5f + 0.5f);
            output.write(reinterpret_cast<const char*>(&byte), 1);
        }
    }
}

struct Options {
    std::filesystem::path modelDirectory = "./data/onnx/sd15_denoiser_256";
    std::filesystem::path outputPath = "build/TestingOutput/sd15_pipeline_klartraum.ppm";
    uint32_t imageSize = 256;
};

Options parseOptions(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--model-dir" && index + 1 < argc) {
            options.modelDirectory = argv[++index];
        } else if (argument == "--output" && index + 1 < argc) {
            options.outputPath = argv[++index];
        } else if (argument == "--size" && index + 1 < argc) {
            options.imageSize = static_cast<uint32_t>(std::stoul(argv[++index]));
        } else {
            throw std::runtime_error("Unknown or incomplete argument: " + argument);
        }
    }
    return options;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parseOptions(argc, argv);
        if (options.imageSize == 0 || options.imageSize % 8 != 0) {
            throw std::runtime_error("--size must be a positive multiple of 8");
        }
        const uint32_t imageSize = options.imageSize;
        const uint32_t latentSize = imageSize / 8;
        const size_t latentElements = 4 * latentSize * latentSize;
        const size_t sampleElements = 2 * latentElements;
        constexpr size_t embeddingElements = 2 * 77 * 768;
        const size_t imageElements = 3 * static_cast<size_t>(imageSize) * imageSize;
        constexpr float guidanceScale = 7.5f;
        constexpr float vaeScalingFactor = 0.18215f;

        const auto& directory = options.modelDirectory;
        const auto unetPath = directory / "sd15_unet.onnx";
        const auto decoderPath = directory / "sd15_vae_decoder.onnx";
        for (const auto& path : {unetPath, decoderPath}) {
            if (!std::filesystem::exists(path)) {
                throw std::runtime_error("Missing " + path.string() +
                                         "; run scripts/sd15_onnx/export_denoiser.py first");
            }
        }

        const auto timesteps = readTensor<int64_t>(directory / "scheduler_timesteps_i64.bin");
        const auto alphaPairs = readTensor<float>(directory / "scheduler_alphas_f32.bin");
        if (timesteps.empty() || alphaPairs.size() != timesteps.size() * 2) {
            throw std::runtime_error("Invalid DDIM scheduler fixtures");
        }
        const auto embeddingsValues = readTensor<float>(directory / "prompt_embeddings_f32.bin", embeddingElements);
        std::vector<float> latents = readTensor<float>(directory / "initial_latents_f32.bin", latentElements);

        klartraum::HeadlessFrontend frontend;
        auto& context = frontend.getKlartraumEngine().getVulkanContext();
        {
            auto network = context.create<klartraum::OnnxNetwork>(unetPath.string());
            const auto& memory = network->getMemoryPlanStats();
            std::cout << "UNet transient storage: " << memory.logicalBytes / (1024.0 * 1024.0) << " logical MiB -> "
                      << memory.allocatedBytes / (1024.0 * 1024.0) << " allocated MiB in " << memory.slotCount
                      << " slots" << std::endl;

            auto sample =
                context.create<klartraum::TensorElement<float>>(std::vector<uint32_t>{2, 4, latentSize, latentSize});
            auto timestep = context.create<klartraum::TensorElement<int64_t>>(std::vector<uint32_t>{1});
            auto embeddings = context.create<klartraum::TensorElement<float>>(std::vector<uint32_t>{2, 77, 768});
            network->setInputTensor("sample", sample);
            network->setInputTensor("timestep", timestep);
            network->setInputTensor("encoder_hidden_states", embeddings);

            klartraum::ComputeGraph graph(context, 1);
            graph.compileFrom(network);
            embeddings->setData(0, embeddingsValues);

            std::vector<float> batch(sampleElements);
            std::vector<float> prediction(sampleElements);
            std::vector<float> guided(latentElements);
            for (size_t step = 0; step < timesteps.size(); ++step) {
                std::copy(latents.begin(), latents.end(), batch.begin());
                std::copy(latents.begin(), latents.end(), batch.begin() + latentElements);
                sample->setData(0, batch);
                timestep->setData(0, std::vector<int64_t>{timesteps[step]});
                graph.submitAndWait(context.getGraphicsQueue(), 0);

                auto output = std::dynamic_pointer_cast<klartraum::TensorElement<float>>(
                    network->getOutputElement("noise_prediction"));
                output->getDataBuffer(0).memcopyTo(prediction);
                if (step == 0) {
                    const float firstStepError = maximumError(
                        prediction, readTensor<float>(directory / "unet_reference_f32.bin", sampleElements));
                    std::cout << "First UNet prediction maximum error: " << firstStepError << std::endl;
                }
                for (size_t index = 0; index < latentElements; ++index) {
                    guided[index] =
                        prediction[index] + guidanceScale * (prediction[latentElements + index] - prediction[index]);
                }

                const float alpha = alphaPairs[step * 2];
                const float previousAlpha = alphaPairs[step * 2 + 1];
                const float sqrtAlpha = std::sqrt(alpha);
                const float sqrtBeta = std::sqrt(1.0f - alpha);
                const float sqrtPreviousAlpha = std::sqrt(previousAlpha);
                const float sqrtPreviousBeta = std::sqrt(1.0f - previousAlpha);
                for (size_t index = 0; index < latentElements; ++index) {
                    const float predictedOriginal = (latents[index] - sqrtBeta * guided[index]) / sqrtAlpha;
                    latents[index] = sqrtPreviousAlpha * predictedOriginal + sqrtPreviousBeta * guided[index];
                }
                std::cout << "Completed DDIM step " << (step + 1) << "/" << timesteps.size()
                          << " (t=" << timesteps[step] << ", max|noise|=" << maximumMagnitude(guided)
                          << ", max|latent|=" << maximumMagnitude(latents) << ")" << std::endl;
            }
        }

        const float latentError =
            maximumError(latents, readTensor<float>(directory / "final_latents_f32.bin", latentElements));
        for (float& value : latents)
            value /= vaeScalingFactor;

        std::vector<float> decoded;
        {
            auto decoder = context.create<klartraum::OnnxNetwork>(decoderPath.string());
            auto latent =
                context.create<klartraum::TensorElement<float>>(std::vector<uint32_t>{1, 4, latentSize, latentSize});
            decoder->setInputTensor("input", latent);
            klartraum::ComputeGraph graph(context, 1);
            graph.compileFrom(decoder);
            latent->setData(0, latents);
            graph.submitAndWait(context.getGraphicsQueue(), 0);
            auto output =
                std::dynamic_pointer_cast<klartraum::TensorElement<float>>(decoder->getOutputElement("output"));
            decoded.resize(output->getDataElementCount());
            output->getDataBuffer(0).memcopyTo(decoded);
        }

        const float imageError =
            maximumError(decoded, readTensor<float>(directory / "pipeline_reference_f32.bin", imageElements));
        std::cout << "Error versus Python/ONNX Runtime: final latent=" << latentError
                  << ", decoded image=" << imageError << std::endl;
        if (latentError > 2e-1f || imageError > 2e-1f) {
            throw std::runtime_error("Klartraum pipeline output is outside its reference tolerance");
        }
        writePpm(options.outputPath, decoded, imageSize);
        std::cout << "Wrote " << options.outputPath << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "SD 1.5 denoiser example failed: " << error.what() << std::endl;
        return 1;
    }
}
