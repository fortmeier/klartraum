// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Text-to-image and image-to-image with Qwen-Image in Klartraum: the Qwen2.5-VL text encoder (GGUF), the 60-block
// transformer (GGUF) with the Lightning LoRA in 8 flow-matching Euler steps, and the VAE (ONNX graphs exported by
// scripts/qwen_image/export_vae.py for the image size). The stages run one after another, so only one large model is
// on the GPU at a time.
//
//   qwen_image_example --prompt "a bonsai tree, watercolor" --size 512 512 --out bonsai.ppm
//   qwen_image_example --prompt "..." --image rendering.ppm --strength 0.6 --out styled.ppm
//
// Images are binary PPM (P6); the input is used at --size (it must have that size).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/tensorelement.hpp"
#include "klartraum/diffusion/qwen_image_transformer.hpp"
#include "klartraum/diffusion/qwen_text_encoder.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/onnx/onnx_network.hpp"

using namespace klartraum;

namespace {

const std::string kModels = "build/models/qwen-image/";

struct Arguments {
    std::string prompt = "a bonsai tree in a clay pot, watercolor painting";
    uint32_t height = 512, width = 512, steps = 8, seed = 0;
    std::string image, out = "build/TestingOutput/qwen_image/image.ppm";
    float strength = 0.6f;
    std::string transformer = kModels + "transformer/qwen-image-Q4_K_S.gguf";
    std::string text = kModels + "text/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf";
    std::string lora = kModels + "lightning/Qwen-Image-Lightning-8steps-V2.0.gguf";
};

Arguments parse(int argc, char** argv) {
    Arguments a;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc)
                throw std::runtime_error(arg + " needs a value");
            return argv[++i];
        };
        if (arg == "--prompt")
            a.prompt = next();
        else if (arg == "--size") {
            a.height = uint32_t(std::stoul(next()));
            a.width = uint32_t(std::stoul(next()));
        } else if (arg == "--steps")
            a.steps = uint32_t(std::stoul(next()));
        else if (arg == "--seed")
            a.seed = uint32_t(std::stoul(next()));
        else if (arg == "--image")
            a.image = next();
        else if (arg == "--strength")
            a.strength = std::stof(next());
        else if (arg == "--out")
            a.out = next();
        else if (arg == "--transformer")
            a.transformer = next();
        else if (arg == "--text-encoder")
            a.text = next();
        else if (arg == "--lora")
            a.lora = next();
        else
            throw std::runtime_error("unknown argument " + arg);
    }
    if (a.height % 16 || a.width % 16)
        throw std::runtime_error("--size must be a multiple of 16");
    return a;
}

double seconds(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

// An image [3][height][width] in [-1, 1] from a binary PPM.
std::vector<float> readPpm(const std::string& path, uint32_t height, uint32_t width) {
    std::ifstream file(path, std::ios::binary);
    std::string magic;
    uint32_t w = 0, h = 0, maximum = 0;
    file >> magic >> w >> h >> maximum;
    file.get();
    if (magic != "P6" || w != width || h != height || maximum != 255)
        throw std::runtime_error(path + " is not a " + std::to_string(width) + "x" + std::to_string(height) +
                                 " binary PPM");
    std::vector<uint8_t> rgb(size_t(width) * height * 3);
    file.read(reinterpret_cast<char*>(rgb.data()), std::streamsize(rgb.size()));
    std::vector<float> image(rgb.size());
    for (size_t p = 0; p < size_t(width) * height; ++p)
        for (int c = 0; c < 3; ++c)
            image[c * size_t(width) * height + p] = rgb[p * 3 + c] / 127.5f - 1.0f;
    return image;
}

void writePpm(const std::string& path, const std::vector<float>& image, uint32_t height, uint32_t width) {
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    std::ofstream file(path, std::ios::binary);
    file << "P6\n" << width << " " << height << "\n255\n";
    for (size_t p = 0; p < size_t(width) * height; ++p)
        for (int c = 0; c < 3; ++c) {
            const float value = std::clamp(image[c * size_t(width) * height + p], -1.0f, 1.0f);
            file.put(char(uint8_t(std::lround((value + 1.0f) * 127.5f))));
        }
}

// Runs an exported VAE graph once on `input`.
std::vector<float> runVae(VulkanContext& vc, const std::string& path, const std::string& inputName,
                          const std::vector<uint32_t>& shape, const std::vector<float>& input,
                          const std::string& outputName) {
    if (!std::filesystem::exists(path))
        throw std::runtime_error(path + " is missing: run scripts/qwen_image/export_vae.py --size H W");
    auto network = vc.create<OnnxNetwork>(path);
    network->retainTensor(outputName);
    auto tensor = vc.create<TensorElement<float>>(shape);
    network->setInputTensor(inputName, tensor, -1);
    ComputeGraph graph(vc, 1);
    graph.compileFrom(network);
    tensor->getDataBuffer(0).memcopyFrom(input);
    graph.submitAndWait(vc.getGraphicsQueue(), 0);
    auto output = std::dynamic_pointer_cast<TensorElement<float>>(network->getOutputElement(outputName));
    std::vector<float> values(output->getDataElementCount());
    output->getDataBuffer(0).memcopyTo(values);
    return values;
}

// Latent [16][h][w] <-> 2x2 patches [h/2 * w/2][64] in diffusers' _pack_latents order (channel, dy, dx).
std::vector<float> pack(const std::vector<float>& latent, uint32_t h, uint32_t w) {
    std::vector<float> packed(latent.size());
    for (uint32_t c = 0; c < 16; ++c)
        for (uint32_t y = 0; y < h; ++y)
            for (uint32_t x = 0; x < w; ++x)
                packed[((y / 2) * (w / 2) + x / 2) * 64 + c * 4 + (y % 2) * 2 + x % 2] = latent[(c * h + y) * w + x];
    return packed;
}

std::vector<float> unpack(const std::vector<float>& packed, uint32_t h, uint32_t w) {
    std::vector<float> latent(packed.size());
    for (uint32_t c = 0; c < 16; ++c)
        for (uint32_t y = 0; y < h; ++y)
            for (uint32_t x = 0; x < w; ++x)
                latent[(c * h + y) * w + x] = packed[((y / 2) * (w / 2) + x / 2) * 64 + c * 4 + (y % 2) * 2 + x % 2];
    return latent;
}

// Lightning's schedule: sigmas evenly from 1 to 1 / steps, exponentially time-shifted with mu = ln 3, then 0.
std::vector<float> lightningSigmas(uint32_t steps) {
    std::vector<float> sigmas;
    for (uint32_t i = 0; i < steps; ++i) {
        const double sigma = steps == 1 ? 1.0 : 1.0 - (1.0 - 1.0 / steps) * double(i) / double(steps - 1);
        sigmas.push_back(float(3.0 / (3.0 + (1.0 / sigma - 1.0))));
    }
    sigmas.push_back(0.0f);
    return sigmas;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Arguments args = parse(argc, argv);
        const uint32_t h = args.height / 8, w = args.width / 8;
        const std::string size = std::to_string(args.height) + "x" + std::to_string(args.width);
        HeadlessFrontend frontend;
        auto& vc = frontend.getKlartraumEngine().getVulkanContext();
        auto start = std::chrono::steady_clock::now();

        std::vector<float> embedding;
        uint32_t textTokens = 0;
        {
            auto encoder = vc.create<QwenTextEncoder>(args.text);
            ComputeGraph graph(vc, 1);
            graph.compileFrom(encoder);
            const auto tokens = encoder->promptTokens(args.prompt);
            encoder->setTokens(tokens);
            graph.submitAndWait(vc.getGraphicsQueue(), 0);
            embedding = encoder->readPromptEmbedding();
            textTokens = uint32_t(tokens.size()) - QwenTextEncoder::kTemplateTokens;
        }
        std::cout << "prompt: " << textTokens << " tokens in " << seconds(start) << " s" << std::endl;

        auto sigmas = lightningSigmas(args.steps);
        std::mt19937 rng(args.seed);
        std::normal_distribution<float> normal;
        std::vector<float> noise(size_t(16) * h * w);
        for (auto& n : noise)
            n = normal(rng);
        std::vector<float> latent = noise;
        uint32_t first = 0;
        if (!args.image.empty()) {
            // Image to image: start from the input's latent noised to the schedule's sigma after skipping the first
            // steps (diffusers' Img2Img pipeline: int(steps - steps * strength) steps skipped).
            const auto clean =
                runVae(vc, kModels + "onnx/vae_encoder_" + size + ".onnx", "image", {1, 3, args.height, args.width},
                       readPpm(args.image, args.height, args.width), "latent");
            first = uint32_t(std::max(0.0, double(args.steps) - double(args.steps) * args.strength));
            const float sigma = sigmas[first];
            for (size_t i = 0; i < latent.size(); ++i)
                latent[i] = sigma * noise[i] + (1.0f - sigma) * clean[i];
        }

        auto packed = pack(latent, h, w);
        {
            auto transformer = vc.create<QwenImageTransformer>(
                args.transformer, QwenImageTransformerOptions{h, w, textTokens, args.lora, 0});
            ComputeGraph graph(vc, 1);
            graph.compileFrom(transformer);
            transformer->setPromptEmbedding(embedding);
            for (uint32_t step = first; step < args.steps; ++step) {
                const auto stepStart = std::chrono::steady_clock::now();
                transformer->setLatent(packed);
                transformer->setSigma(sigmas[step]);
                graph.submitAndWait(vc.getGraphicsQueue(), 0);
                const auto velocity = transformer->readVelocity();
                for (size_t i = 0; i < packed.size(); ++i)
                    packed[i] += (sigmas[step + 1] - sigmas[step]) * velocity[i];
                std::cout << "step " << step + 1 << "/" << args.steps << " (sigma " << sigmas[step] << ") in "
                          << seconds(stepStart) << " s" << std::endl;
            }
        }

        const auto image = runVae(vc, kModels + "onnx/vae_decoder_" + size + ".onnx", "latent", {1, 16, h, w},
                                  unpack(packed, h, w), "image");
        writePpm(args.out, image, args.height, args.width);
        std::cout << "wrote " << args.out << " after " << seconds(start) << " s" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "qwen_image_example: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
