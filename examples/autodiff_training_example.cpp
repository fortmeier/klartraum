// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Trains a small network on the GPU to reproduce an image from pixel coordinates. The network is built from
// the existing layers (MatMul, bias Add, ReLU) by training::MlpTraining; the gradients of the non-linear parts
// come from Slang's automatic differentiation (shaders/training/differentiable.slang). Every training step is
// one pre-recorded compute graph: forward pass, loss gradient, backward pass and Adam updates.
//
// Usage: autodiff_training_example [--steps N] [--size S]
// Writes the target and snapshots of the prediction to build/TestingOutput/autodiff_training/.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "klartraum/headless_frontend.hpp"
#include "klartraum/layers/mlp_training.hpp"

namespace {

constexpr int kFrequencies = 4; // sin and cos of 2^k pi x for k < kFrequencies, per coordinate

// The pattern to learn at (x, y) in [-1, 1]^2: rings around a disc, over a checkerboard, in [0, 1].
float target(float x, float y) {
    const float r = std::sqrt(x * x + y * y);
    if (r < 0.25f)
        return 1.0f;
    if (r < 0.85f)
        return 0.5f + 0.5f * std::cos(18.0f * r);
    const bool checker = (int(std::floor(4.0f * (x + 1.0f))) + int(std::floor(4.0f * (y + 1.0f)))) % 2 == 0;
    return checker ? 0.8f : 0.2f;
}

void writePgmAsPpm(const std::filesystem::path& path, const std::vector<float>& values, uint32_t size) {
    std::ofstream out(path, std::ios::binary);
    out << "P6\n" << size << " " << size << "\n255\n";
    for (float v : values) {
        const auto byte = static_cast<unsigned char>(std::lround(std::clamp(v, 0.0f, 1.0f) * 255.0f));
        out.put(byte).put(byte).put(byte);
    }
}

} // namespace

int main(int argc, char** argv) {
    uint32_t steps = 2000, size = 64;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--steps" && i + 1 < argc)
            steps = uint32_t(std::stoul(argv[++i]));
        else if (argument == "--size" && i + 1 < argc)
            size = uint32_t(std::stoul(argv[++i]));
    }
    const std::filesystem::path outputDirectory = "build/TestingOutput/autodiff_training";
    std::filesystem::create_directories(outputDirectory);

    // Inputs: the coordinates and their Fourier features; targets: the pattern.
    const uint32_t features = 2 + 4 * kFrequencies, pixels = size * size;
    std::vector<float> inputs, targets;
    inputs.reserve(size_t(pixels) * features);
    for (uint32_t py = 0; py < size; ++py) {
        for (uint32_t px = 0; px < size; ++px) {
            const float x = (float(px) + 0.5f) / float(size) * 2.0f - 1.0f;
            const float y = (float(py) + 0.5f) / float(size) * 2.0f - 1.0f;
            inputs.push_back(x);
            inputs.push_back(y);
            for (int k = 0; k < kFrequencies; ++k) {
                const float scale = float(M_PI) * float(1 << k);
                for (float c : {x, y}) {
                    inputs.push_back(std::sin(scale * c));
                    inputs.push_back(std::cos(scale * c));
                }
            }
            targets.push_back(target(x, y));
        }
    }
    writePgmAsPpm(outputDirectory / "target.ppm", targets, size);

    klartraum::HeadlessFrontend frontend;
    auto& vulkanContext = frontend.getKlartraumEngine().getVulkanContext();
    klartraum::training::MlpOptions options;
    options.layerSizes = {features, 64, 64, 1};
    options.batch = pixels; // the whole image every step
    options.adam.learningRate = 0.005f;
    klartraum::training::MlpTraining mlp(vulkanContext, options);
    mlp.setInputs(inputs);
    mlp.setTargets(targets);

    const auto started = std::chrono::steady_clock::now();
    for (uint32_t step = 1; step <= steps; ++step) {
        mlp.step();
        if (step == 1 || step == 10 || step == 100 || step == 500 || step % 1000 == 0 || step == steps) {
            std::cout << "step " << step << ": loss " << mlp.loss() << std::endl;
            writePgmAsPpm(outputDirectory / ("prediction_" + std::to_string(step) + ".ppm"), mlp.outputs(), size);
        }
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    std::cout << steps << " steps in " << seconds << " s (" << seconds / steps * 1e3 << " ms per step); images in "
              << outputDirectory << std::endl;
    return 0;
}
