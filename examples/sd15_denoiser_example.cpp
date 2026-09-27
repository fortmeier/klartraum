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

float maximumError(const std::vector<float>& actual, const std::vector<float>& expected) {
    if (actual.size() != expected.size())
        throw std::runtime_error("Reference tensor size mismatch");
    float result = 0.0f;
    for (size_t index = 0; index < actual.size(); ++index) {
        result = std::max(result, std::abs(actual[index] - expected[index]));
    }
    return result;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const std::filesystem::path directory =
            argc > 1 ? std::filesystem::path(argv[1]) : std::filesystem::path("./data/onnx/sd15_denoiser_128");
        const auto modelPath = directory / "sd15_unet.onnx";
        if (!std::filesystem::exists(modelPath)) {
            throw std::runtime_error("Missing " + modelPath.string() +
                                     "; run scripts/sd15_onnx/export_denoiser.py first");
        }

        constexpr size_t latentElements = 4 * 16 * 16;
        constexpr size_t sampleElements = 2 * latentElements;
        constexpr size_t embeddingElements = 2 * 77 * 768;
        constexpr float guidanceScale = 7.5f;

        klartraum::HeadlessFrontend frontend;
        auto& context = frontend.getKlartraumEngine().getVulkanContext();
        auto network = context.create<klartraum::OnnxNetwork>(modelPath.string());
        const auto& memory = network->getMemoryPlanStats();
        std::cout << "UNet transient storage: " << memory.logicalBytes / (1024.0 * 1024.0) << " logical MiB -> "
                  << memory.allocatedBytes / (1024.0 * 1024.0) << " allocated MiB in " << memory.slotCount << " slots"
                  << std::endl;

        auto sample = context.create<klartraum::TensorElement<float>>(std::vector<uint32_t>{2, 4, 16, 16});
        auto timestep = context.create<klartraum::TensorElement<int64_t>>(std::vector<uint32_t>{1});
        auto embeddings = context.create<klartraum::TensorElement<float>>(std::vector<uint32_t>{2, 77, 768});
        network->setInputTensor("sample", sample);
        network->setInputTensor("timestep", timestep);
        network->setInputTensor("encoder_hidden_states", embeddings);

        klartraum::ComputeGraph graph(context, 1);
        graph.compileFrom(network);
        sample->setData(0, readTensor<float>(directory / "unet_sample_f32.bin", sampleElements));
        timestep->setData(0, readTensor<int64_t>(directory / "unet_timestep_i64.bin", 1));
        embeddings->setData(0, readTensor<float>(directory / "prompt_embeddings_f32.bin", embeddingElements));
        graph.submitAndWait(context.getGraphicsQueue(), 0);

        auto output =
            std::dynamic_pointer_cast<klartraum::TensorElement<float>>(network->getOutputElement("noise_prediction"));
        std::vector<float> prediction(output->getDataElementCount());
        output->getDataBuffer(0).memcopyTo(prediction);
        const float predictionError =
            maximumError(prediction, readTensor<float>(directory / "unet_reference_f32.bin", sampleElements));

        std::vector<float> guided(latentElements);
        for (size_t index = 0; index < latentElements; ++index) {
            guided[index] =
                prediction[index] + guidanceScale * (prediction[latentElements + index] - prediction[index]);
        }
        const float guidanceError =
            maximumError(guided, readTensor<float>(directory / "guided_noise_f32.bin", latentElements));
        std::cout << "Error versus ONNX Runtime: UNet=" << predictionError
                  << ", classifier-free guidance=" << guidanceError << std::endl;
        if (predictionError > 2e-2f || guidanceError > 1e-1f) {
            throw std::runtime_error("Klartraum denoiser output is outside its reference tolerance");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "SD 1.5 denoiser example failed: " << error.what() << std::endl;
        return 1;
    }
}
