// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// NVIDIA Cosmos3-Edge image-to-video on Klartraum.
//
// Runs the graphs exported by scripts/cosmos3/export_onnx.py one stage at a
// time so that only one model is resident: the text tower (once per prompt
// pair), then the video denoiser (once per scheduler step), then the Wan VAE.
// Every stage is checked against the Python fixtures exported with the graphs;
// with --skip-checks only the inputs written by `export_onnx.py prepare` are read.

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/copybuffer.hpp"
#include "klartraum/computegraph/noop.hpp"
#include "klartraum/cosmos3/conditioning_image.hpp"
#include "klartraum/cosmos3/tiled_decode.hpp"
#include "klartraum/cosmos3/unipc_flow_scheduler.hpp"
#include "klartraum/computegraph/tensorelement.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/onnx/onnx_network.hpp"

namespace {

namespace fs = std::filesystem;

template <typename T>
std::vector<T> readTensor(const fs::path& path, size_t count) {
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

/** Numeric entries of config.txt ("key value..." per line). */
class Config {
public:
    explicit Config(const fs::path& path) {
        std::ifstream input(path);
        if (!input)
            throw std::runtime_error("Missing " + path.string() + "; run scripts/cosmos3/export_onnx.py");
        std::string line;
        while (std::getline(input, line)) {
            std::istringstream stream(line);
            std::string key;
            stream >> key;
            double value;
            while (stream >> value)
                values[key].push_back(value);
        }
    }
    double number(const std::string& key, size_t index = 0) const {
        const auto it = values.find(key);
        if (it == values.end() || index >= it->second.size())
            throw std::runtime_error("config.txt lacks " + key);
        return it->second[index];
    }
    uint32_t count(const std::string& key, size_t index = 0) const { return static_cast<uint32_t>(number(key, index)); }
    /** @p key's first value, or @p fallback if config.txt lacks it. */
    uint32_t countOr(const std::string& key, uint32_t fallback) const {
        return values.count(key) ? count(key) : fallback;
    }

private:
    std::map<std::string, std::vector<double>> values;
};

struct Comparison {
    float maxError = 0.0f;
    float maxReference = 0.0f;
};

Comparison compare(const std::vector<float>& actual, const std::vector<float>& expected) {
    if (actual.size() != expected.size())
        throw std::runtime_error("Reference tensor size mismatch");
    Comparison result;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]))
            return {std::numeric_limits<float>::infinity(), 0.0f};
        result.maxError = std::max(result.maxError, std::abs(actual[i] - expected[i]));
        result.maxReference = std::max(result.maxReference, std::abs(expected[i]));
    }
    return result;
}

struct Options {
    fs::path modelDirectory = "./data/onnx/cosmos3_256";
    fs::path outputDirectory = "build/TestingOutput/cosmos3";
    fs::path image; ///< conditioning PPM; empty: the exported image_f32.bin
    std::string name = "lantern_orbit";
    std::string stage = "all"; ///< all, text, step, vae
    size_t maxSteps = std::numeric_limits<size_t>::max();
    bool profile = false;
    bool check = true; ///< compare stages with the float32 reference fixtures
};

Options parseOptions(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--model-dir" && index + 1 < argc) {
            options.modelDirectory = argv[++index];
        } else if (argument == "--output-dir" && index + 1 < argc) {
            options.outputDirectory = argv[++index];
        } else if (argument == "--image" && index + 1 < argc) {
            options.image = argv[++index];
        } else if (argument == "--name" && index + 1 < argc) {
            options.name = argv[++index];
        } else if (argument == "--stage" && index + 1 < argc) {
            options.stage = argv[++index];
        } else if (argument == "--max-steps" && index + 1 < argc) {
            options.maxSteps = std::stoul(argv[++index]);
        } else if (argument == "--profile") {
            options.profile = true;
        } else if (argument == "--skip-checks") {
            options.check = false;
        } else {
            throw std::runtime_error("Unknown or incomplete argument: " + argument);
        }
    }
    return options;
}

void printProfiling(const std::string& stage, const klartraum::ComputeGraph& graph) {
    auto results = graph.getProfilingResults();
    std::map<std::string, double> totals;
    double total = 0.0;
    for (const auto& [name, milliseconds] : results) {
        totals[name.substr(0, name.find('_'))] += milliseconds;
        total += milliseconds;
    }
    std::vector<std::pair<std::string, double>> sorted(totals.begin(), totals.end());
    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    std::cout << stage << " GPU profile: " << total << " ms across " << results.size() << " dispatches\n ";
    for (size_t i = 0; i < std::min<size_t>(10, sorted.size()); ++i) {
        std::cout << " " << sorted[i].first << "=" << sorted[i].second << " ms";
    }
    std::cout << std::endl;
    std::vector<std::pair<std::string, double>> slowest(results.begin(), results.end());
    std::sort(slowest.begin(), slowest.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    std::cout << "  slowest dispatches:";
    for (size_t i = 0; i < std::min<size_t>(5, slowest.size()); ++i) {
        std::cout << "\n    " << slowest[i].first << " " << slowest[i].second << " ms";
    }
    // Totals per node kind: the name without digits, so all layers of one projection add up.
    std::map<std::string, std::pair<double, size_t>> kinds;
    for (const auto& [name, milliseconds] : results) {
        std::string kind;
        for (char c : name) {
            if (!std::isdigit(static_cast<unsigned char>(c)))
                kind += c;
        }
        kinds[kind].first += milliseconds;
        ++kinds[kind].second;
    }
    std::vector<std::pair<std::string, std::pair<double, size_t>>> byKind(kinds.begin(), kinds.end());
    std::sort(byKind.begin(), byKind.end(),
              [](const auto& a, const auto& b) { return a.second.first > b.second.first; });
    std::cout << "\n  by node kind:";
    for (size_t i = 0; i < std::min<size_t>(15, byKind.size()); ++i) {
        std::cout << "\n    " << byKind[i].first << " " << byKind[i].second.first << " ms (" << byKind[i].second.second
                  << "x)";
    }
    std::cout << std::endl;
}

double secondsSince(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

constexpr VkBufferUsageFlags kInputUsage =
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

/** Per-layer text keys [2, KV, D, L] and values [2, KV, L, D] for both guidance prompts. */
struct TextKeyValues {
    std::vector<std::vector<float>> keys;
    std::vector<std::vector<float>> values;
};

TextKeyValues runTextTower(klartraum::VulkanContext& context, const Config& config, const Options& options) {
    const auto& directory = options.modelDirectory;
    const uint32_t textLength = config.count("text_length");
    const uint32_t headDim = config.count("head_dim");
    const uint32_t kvHeads = config.count("kv_heads");
    const uint32_t layers = config.count("num_layers");
    const size_t ropeElements = size_t(2) * textLength * headDim;
    const size_t kvElements = size_t(2) * kvHeads * textLength * headDim;

    const auto loadStarted = std::chrono::steady_clock::now();
    auto network = context.create<klartraum::OnnxNetwork>((directory / "text_kv.onnx").string());
    auto inputIds =
        context.create<klartraum::TensorElement<int64_t>>(std::vector<uint32_t>{2, textLength}, kInputUsage);
    auto ropeCos =
        context.create<klartraum::TensorElement<float>>(std::vector<uint32_t>{2, textLength, headDim}, kInputUsage);
    auto ropeSin =
        context.create<klartraum::TensorElement<float>>(std::vector<uint32_t>{2, textLength, headDim}, kInputUsage);
    network->setInputTensor("input_ids", inputIds);
    network->setInputTensor("rope_cos", ropeCos);
    network->setInputTensor("rope_sin", ropeSin);
    for (uint32_t layer = 0; layer < layers; ++layer) {
        network->retainTensor("text_k_" + std::to_string(layer));
        network->retainTensor("text_v_" + std::to_string(layer));
    }
    klartraum::ComputeGraph graph(context, 1);
    if (options.profile)
        graph.enableProfiling();
    graph.compileFrom(network);
    std::cout << "Text tower loaded in " << secondsSince(loadStarted) << " s" << std::endl;

    inputIds->setData(0, readTensor<int64_t>(directory / "input_ids_i64.bin", size_t(2) * textLength));
    ropeCos->setData(0, readTensor<float>(directory / "text_rope_cos_f32.bin", ropeElements));
    ropeSin->setData(0, readTensor<float>(directory / "text_rope_sin_f32.bin", ropeElements));
    const auto started = std::chrono::steady_clock::now();
    graph.submitAndWait(context.getGraphicsQueue(), 0);
    std::cout << "Klartraum text tower: " << secondsSince(started) << " s" << std::endl;
    if (options.profile)
        printProfiling("Text tower", graph);

    TextKeyValues result;
    float worstRelative = 0.0f;
    for (uint32_t layer = 0; layer < layers; ++layer) {
        for (const char* kind : {"k", "v"}) {
            const std::string name = std::string("text_") + kind + "_" + std::to_string(layer);
            auto output = std::dynamic_pointer_cast<klartraum::TensorElement<float>>(network->getOutputElement(name));
            if (!output || output->getDataElementCount() != kvElements) {
                throw std::runtime_error("Unexpected text tower output " + name);
            }
            std::vector<float> values(kvElements);
            output->getDataBuffer(0).memcopyTo(values);
            const auto fixture = directory / (name + "_f32.bin");
            if (fs::exists(fixture)) {
                const auto check = compare(values, readTensor<float>(fixture, kvElements));
                worstRelative = std::max(worstRelative, check.maxError / std::max(check.maxReference, 1e-6f));
                if (layer == 0 || layer == layers - 1 || !std::isfinite(check.maxError)) {
                    std::cout << "  " << name << ": max error " << check.maxError << " (max |ref| "
                              << check.maxReference << ")" << std::endl;
                }
            }
            (kind[0] == 'k' ? result.keys : result.values).push_back(std::move(values));
        }
    }
    std::cout << "Text K/V worst relative error: " << worstRelative << std::endl;
    if (!(worstRelative < 1e-3f))
        throw std::runtime_error("Text tower is outside its reference tolerance");
    return result;
}

/** Latent geometry shared by the host-side packing helpers. */
struct LatentShape {
    uint32_t channels, frames, height, width, patch;
    size_t elements() const { return size_t(channels) * frames * height * width; }
    uint32_t tokensPerFrame() const { return (height / patch) * (width / patch); }
    size_t tokens() const { return size_t(frames) * tokensPerFrame(); }
    uint32_t tokenWidth() const { return patch * patch * channels; }
};

/** [C, T, H, W] -> [T * H/p * W/p, p * p * C], token (t, y, x), feature (py, px, c). */
std::vector<float> patchify(const std::vector<float>& latents, const LatentShape& s) {
    std::vector<float> tokens(latents.size());
    const uint32_t gh = s.height / s.patch, gw = s.width / s.patch;
    for (uint32_t c = 0; c < s.channels; ++c)
        for (uint32_t t = 0; t < s.frames; ++t)
            for (uint32_t y = 0; y < s.height; ++y)
                for (uint32_t x = 0; x < s.width; ++x) {
                    const size_t token = (size_t(t) * gh + y / s.patch) * gw + x / s.patch;
                    const size_t feature = (size_t(y % s.patch) * s.patch + x % s.patch) * s.channels + c;
                    tokens[token * s.tokenWidth() + feature] =
                        latents[((size_t(c) * s.frames + t) * s.height + y) * s.width + x];
                }
    return tokens;
}

/** Inverse of patchify. */
std::vector<float> unpatchify(const float* tokens, const LatentShape& s) {
    std::vector<float> latents(s.elements());
    const uint32_t gh = s.height / s.patch, gw = s.width / s.patch;
    for (uint32_t c = 0; c < s.channels; ++c)
        for (uint32_t t = 0; t < s.frames; ++t)
            for (uint32_t y = 0; y < s.height; ++y)
                for (uint32_t x = 0; x < s.width; ++x) {
                    const size_t token = (size_t(t) * gh + y / s.patch) * gw + x / s.patch;
                    const size_t feature = (size_t(y % s.patch) * s.patch + x % s.patch) * s.channels + c;
                    latents[((size_t(c) * s.frames + t) * s.height + y) * s.width + x] =
                        tokens[token * s.tokenWidth() + feature];
                }
    return latents;
}

/** The denoiser graph with its inputs bound; one submit evaluates both guidance branches. */
class Denoiser {
public:
    Denoiser(klartraum::VulkanContext& context, const Config& config, const Options& options, const LatentShape& shape,
             const TextKeyValues& text)
        : context(context),
          shape(shape),
          graph(context, 1) {
        const auto& directory = options.modelDirectory;
        const uint32_t textLength = config.count("text_length");
        const uint32_t headDim = config.count("head_dim");
        const uint32_t kvHeads = config.count("kv_heads");
        const uint32_t tokens = static_cast<uint32_t>(shape.tokens());
        const auto started = std::chrono::steady_clock::now();
        network = context.create<klartraum::OnnxNetwork>((directory / "denoiser.onnx").string());
        auto input = [&](const std::string& name, const std::vector<uint32_t>& dims) {
            auto tensor = context.create<klartraum::TensorElement<float>>(dims, kInputUsage);
            network->setInputTensor(name, tensor);
            return tensor;
        };
        tokensInput = input("tokens", {tokens, shape.tokenWidth()});
        timestep = input("timestep", {1});
        auto noisyMask = input("noisy_mask", {tokens, 1});
        auto ropeCos = input("rope_cos", {2, tokens, headDim});
        auto ropeSin = input("rope_sin", {2, tokens, headDim});
        auto textBias = input("text_bias", {2, 1, 1, textLength + tokens});
        std::vector<std::shared_ptr<klartraum::TensorElement<float>>> keyValues;
        for (size_t layer = 0; layer < text.keys.size(); ++layer) {
            keyValues.push_back(input("text_k_" + std::to_string(layer), {2, kvHeads, headDim, textLength}));
            keyValues.push_back(input("text_v_" + std::to_string(layer), {2, kvHeads, textLength, headDim}));
        }
        if (options.profile)
            graph.enableProfiling();
        graph.compileFrom(network);

        const size_t ropeElements = size_t(2) * tokens * headDim;
        std::vector<float> mask(tokens, 1.0f);
        std::fill(mask.begin(), mask.begin() + shape.tokensPerFrame(), 0.0f);
        noisyMask->setData(0, mask);
        ropeCos->setData(0, readTensor<float>(directory / "vision_rope_cos_f32.bin", ropeElements));
        ropeSin->setData(0, readTensor<float>(directory / "vision_rope_sin_f32.bin", ropeElements));
        textBias->setData(0, readTensor<float>(directory / "text_bias_f32.bin", size_t(2) * (textLength + tokens)));
        for (size_t layer = 0; layer < text.keys.size(); ++layer) {
            keyValues[2 * layer]->setData(0, text.keys[layer]);
            keyValues[2 * layer + 1]->setData(0, text.values[layer]);
        }
        std::cout << "Denoiser loaded in " << secondsSince(started) << " s" << std::endl;
    }

    /** Velocity tokens [2, N, p*p*C] (conditional, unconditional) for @p latents at @p t. */
    std::vector<float> predict(const std::vector<float>& latents, float t) {
        tokensInput->setData(0, patchify(latents, shape));
        timestep->setData(0, std::vector<float>{t});
        graph.submitAndWait(context.getGraphicsQueue(), 0);
        auto output = std::dynamic_pointer_cast<klartraum::TensorElement<float>>(network->getOutputElement("velocity"));
        std::vector<float> velocity(output->getDataElementCount());
        output->getDataBuffer(0).memcopyTo(velocity);
        return velocity;
    }

    const klartraum::ComputeGraph& computeGraph() const { return graph; }

private:
    klartraum::VulkanContext& context;
    LatentShape shape;
    std::shared_ptr<klartraum::OnnxNetwork> network;
    std::shared_ptr<klartraum::TensorElement<float>> tokensInput, timestep;
    klartraum::ComputeGraph graph;
};

/** Runs a single-input, single-output graph once and returns its output. */
std::vector<float> runOnce(klartraum::VulkanContext& context, const Options& options, const std::string& model,
                           const std::string& inputName, const std::vector<uint32_t>& inputShape,
                           const std::vector<float>& input, const std::string& outputName, const std::string& label) {
    const auto loadStarted = std::chrono::steady_clock::now();
    auto network = context.create<klartraum::OnnxNetwork>((options.modelDirectory / model).string());
    auto tensor = context.create<klartraum::TensorElement<float>>(inputShape, kInputUsage);
    network->setInputTensor(inputName, tensor);
    klartraum::ComputeGraph graph(context, 1);
    if (options.profile)
        graph.enableProfiling();
    graph.compileFrom(network);
    const auto& memory = network->getMemoryPlanStats();
    std::cout << label << " loaded in " << secondsSince(loadStarted) << " s, transient storage "
              << memory.allocatedBytes / (1024.0 * 1024.0) << " MiB" << std::endl;
    tensor->setData(0, input);
    const auto started = std::chrono::steady_clock::now();
    graph.submitAndWait(context.getGraphicsQueue(), 0);
    std::cout << "Klartraum " << label << ": " << secondsSince(started) << " s" << std::endl;
    if (options.profile)
        printProfiling(label, graph);
    auto output = std::dynamic_pointer_cast<klartraum::TensorElement<float>>(network->getOutputElement(outputName));
    if (!output)
        throw std::runtime_error(label + " has no float output " + outputName);
    std::vector<float> result(output->getDataElementCount());
    output->getDataBuffer(0).memcopyTo(result);
    return result;
}

/**
 * Decodes [C, T, H, W] latents in overlapping square tiles with a decoder
 * exported for @p tile x @p tile latents; the graph is compiled once and
 * submitted per tile, so only one tile's activations are resident.
 */
std::vector<float> decodeTiled(klartraum::VulkanContext& context, const Options& options, const LatentShape& shape,
                               const std::vector<float>& latents, uint32_t tile, uint32_t stride) {
    const auto loadStarted = std::chrono::steady_clock::now();
    auto network = context.create<klartraum::OnnxNetwork>((options.modelDirectory / "vae_decoder.onnx").string());
    auto input = context.create<klartraum::TensorElement<float>>(
        std::vector<uint32_t>{1, shape.channels, shape.frames, tile, tile}, kInputUsage);
    network->setInputTensor("latents", input);
    klartraum::ComputeGraph graph(context, 1);
    if (options.profile)
        graph.enableProfiling();
    graph.compileFrom(network);
    auto output = std::dynamic_pointer_cast<klartraum::TensorElement<float>>(network->getOutputElement("video"));
    if (!output)
        throw std::runtime_error("VAE decoder has no float output video");
    const auto& memory = network->getMemoryPlanStats();
    std::cout << "VAE decoder (" << tile << "x" << tile << " latent tiles) loaded in " << secondsSince(loadStarted)
              << " s, transient storage " << memory.allocatedBytes / (1024.0 * 1024.0) << " MiB" << std::endl;

    std::vector<float> tileLatents(size_t(shape.channels) * shape.frames * tile * tile);
    std::vector<float> tileVideo(output->getDataElementCount());
    const uint32_t scale = 16; // Wan2.2 VAE spatial compression
    const size_t tilePixels = size_t(tile) * scale * tile * scale;
    const uint32_t frames = uint32_t(tileVideo.size() / (3 * tilePixels));
    klartraum::TileBlender blender(3, frames, shape.height * scale, shape.width * scale, (tile - stride) * scale);
    for (uint32_t top : klartraum::tileOffsets(shape.height, tile, stride)) {
        for (uint32_t left : klartraum::tileOffsets(shape.width, tile, stride)) {
            for (uint32_t ct = 0; ct < shape.channels * shape.frames; ++ct) {
                for (uint32_t y = 0; y < tile; ++y) {
                    std::copy_n(latents.begin() + (size_t(ct) * shape.height + top + y) * shape.width + left, tile,
                                tileLatents.begin() + (size_t(ct) * tile + y) * tile);
                }
            }
            input->setData(0, tileLatents);
            const auto started = std::chrono::steady_clock::now();
            graph.submitAndWait(context.getGraphicsQueue(), 0);
            output->getDataBuffer(0).memcopyTo(tileVideo);
            blender.add(tileVideo, tile * scale, tile * scale, top * scale, left * scale);
            std::cout << "  tile (" << top << ", " << left << ") in " << secondsSince(started) << " s" << std::endl;
        }
    }
    std::cout << "Klartraum VAE decoder (tiled): " << secondsSince(loadStarted) << " s" << std::endl;
    if (options.profile)
        printProfiling("VAE decoder", graph);
    return blender.result();
}

/**
 * Copies a cache output back into its cache input once the whole network has
 * run: input 2 is the network itself, which orders the copy after all of it.
 */
class CacheHandoff : public klartraum::CopyBuffer {
public:
    using klartraum::CopyBuffer::CopyBuffer;
    void checkInput(klartraum::ComputeGraphElementPtr input, int index) override {
        if (index != 2)
            klartraum::CopyBuffer::checkInput(input, index);
    }
};

/**
 * Decodes [C, T, H, W] latents one latent frame at a time: the first-chunk
 * decoder yields frame 0, the per-chunk decoder four frames per further latent
 * frame. Both pass the last input frames of every causal convolution on as
 * caches, so only one chunk's activations are resident. The per-chunk graph
 * copies its cache outputs into its cache inputs on the GPU.
 */
std::vector<float> decodeChunked(klartraum::VulkanContext& context, const Options& options, const LatentShape& shape,
                                 const std::vector<float>& latents, uint32_t cacheCount, uint32_t height,
                                 uint32_t width) {
    const auto started = std::chrono::steady_clock::now();
    const size_t plane = size_t(height) * width;
    const uint32_t frames = 4 * shape.frames - 3;
    std::vector<float> video(size_t(3) * frames * plane);
    const std::vector<uint32_t> latentShape{1, shape.channels, 1, shape.height, shape.width};
    std::vector<float> latentFrame(size_t(shape.channels) * shape.height * shape.width);
    auto selectLatentFrame = [&](uint32_t t) {
        for (uint32_t c = 0; c < shape.channels; ++c) {
            std::copy_n(latents.begin() + (size_t(c) * shape.frames + t) * shape.height * shape.width,
                        size_t(shape.height) * shape.width,
                        latentFrame.begin() + size_t(c) * shape.height * shape.width);
        }
    };
    // Copies a chunk's [1, 3, n, H, W] frames to frame `first` of the [3, frames, H, W] video.
    auto placeFrames = [&](const std::vector<float>& chunk, uint32_t first) {
        const uint32_t n = uint32_t(chunk.size() / (3 * plane));
        for (uint32_t c = 0; c < 3; ++c) {
            std::copy_n(chunk.begin() + size_t(c) * n * plane, n * plane,
                        video.begin() + (size_t(c) * frames + first) * plane);
        }
    };
    auto outputOf = [](const std::shared_ptr<klartraum::OnnxNetwork>& network, const std::string& name) {
        auto output = std::dynamic_pointer_cast<klartraum::TensorElement<float>>(network->getOutputElement(name));
        if (!output)
            throw std::runtime_error("VAE chunk decoder has no float output " + name);
        return output;
    };

    std::vector<std::vector<float>> caches;
    std::vector<std::vector<uint32_t>> cacheShapes;
    {
        auto network =
            context.create<klartraum::OnnxNetwork>((options.modelDirectory / "vae_decoder_first.onnx").string());
        auto input = context.create<klartraum::TensorElement<float>>(latentShape, kInputUsage);
        network->setInputTensor("latents", input);
        klartraum::ComputeGraph graph(context, 1);
        graph.compileFrom(network);
        // Without a count from config.txt, every cache_out_<i> the first-chunk graph produces.
        if (cacheCount == 0) {
            try {
                while (network->getOutputElement("cache_out_" + std::to_string(cacheCount)))
                    ++cacheCount;
            } catch (const std::runtime_error&) {
                // getOutputElement throws past the last cache output.
            }
        }
        caches.resize(cacheCount);
        cacheShapes.resize(cacheCount);
        selectLatentFrame(0);
        input->setData(0, latentFrame);
        graph.submitAndWait(context.getGraphicsQueue(), 0);
        auto output = outputOf(network, "video");
        std::vector<float> chunk(output->getDataElementCount());
        output->getDataBuffer(0).memcopyTo(chunk);
        placeFrames(chunk, 0);
        for (uint32_t i = 0; i < cacheCount; ++i) {
            auto cache = outputOf(network, "cache_out_" + std::to_string(i));
            caches[i].resize(cache->getDataElementCount());
            cache->getDataBuffer(0).memcopyTo(caches[i]);
            cacheShapes[i] = cache->getDimensions();
        }
    }
    size_t cacheBytes = 0;
    for (const auto& cache : caches)
        cacheBytes += cache.size() * sizeof(float);
    std::cout << "VAE decoder first chunk in " << secondsSince(started) << " s, " << cacheCount << " caches of "
              << cacheBytes / (1024.0 * 1024.0) << " MiB" << std::endl;

    auto network = context.create<klartraum::OnnxNetwork>((options.modelDirectory / "vae_decoder_chunk.onnx").string());
    auto input = context.create<klartraum::TensorElement<float>>(latentShape, kInputUsage);
    network->setInputTensor("latents", input);
    std::vector<std::shared_ptr<klartraum::TensorElement<float>>> cacheInputs;
    for (uint32_t i = 0; i < cacheCount; ++i) {
        cacheInputs.push_back(context.create<klartraum::TensorElement<float>>(cacheShapes[i], kInputUsage));
        network->setInputTensor("cache_in_" + std::to_string(i), cacheInputs.back());
    }
    auto sink = context.create<klartraum::NoOp>();
    sink->setInput(network, 0);
    for (uint32_t i = 0; i < cacheCount; ++i) {
        auto handoff = context.create<CacheHandoff>();
        handoff->setName("cache_handoff_" + std::to_string(i));
        handoff->setSrcIndex(0);
        handoff->setDstIndex(1);
        handoff->setInput(outputOf(network, "cache_out_" + std::to_string(i)), 0);
        handoff->setInput(cacheInputs[i], 1);
        handoff->setInput(network, 2);
        sink->setInput(handoff, int(i) + 1);
    }
    klartraum::ComputeGraph graph(context, 1);
    if (options.profile)
        graph.enableProfiling();
    graph.compileFrom(sink);
    for (uint32_t i = 0; i < cacheCount; ++i)
        cacheInputs[i]->setData(0, caches[i]);
    std::cout << "VAE chunk decoder loaded, transient storage "
              << network->getMemoryPlanStats().allocatedBytes / (1024.0 * 1024.0) << " MiB" << std::endl;
    auto output = outputOf(network, "video");
    std::vector<float> chunk(output->getDataElementCount());
    for (uint32_t t = 1; t < shape.frames; ++t) {
        const auto chunkStarted = std::chrono::steady_clock::now();
        selectLatentFrame(t);
        input->setData(0, latentFrame);
        graph.submitAndWait(context.getGraphicsQueue(), 0);
        output->getDataBuffer(0).memcopyTo(chunk);
        placeFrames(chunk, 4 * t - 3);
        std::cout << "  latent frame " << t << " in " << secondsSince(chunkStarted) << " s" << std::endl;
    }
    std::cout << "Klartraum VAE decoder (chunked): " << secondsSince(started) << " s" << std::endl;
    if (options.profile)
        printProfiling("VAE chunk decoder", graph);
    return video;
}

void requireClose(const std::string& label, const Comparison& check, float relativeTolerance) {
    std::cout << label << ": max error " << check.maxError << " (max |ref| " << check.maxReference << ")" << std::endl;
    if (!(check.maxError <= relativeTolerance * std::max(check.maxReference, 1e-6f))) {
        throw std::runtime_error(label + " is outside its reference tolerance");
    }
}

/** Writes [3, T, H, W] frames in [-1, 1] as PPM images and one 4:4:4 YUV4MPEG2 video. */
void writeVideo(const fs::path& directory, const std::string& name, const std::vector<float>& video, uint32_t frames,
                uint32_t height, uint32_t width, uint32_t fps) {
    fs::create_directories(directory);
    const size_t plane = size_t(height) * width;
    const size_t channelStride = size_t(frames) * plane;
    auto byte = [](float value) {
        return static_cast<unsigned char>(std::clamp((value + 1.0f) * 127.5f + 0.5f, 0.0f, 255.0f));
    };
    std::ofstream y4m(directory / (name + "_klartraum.y4m"), std::ios::binary);
    y4m << "YUV4MPEG2 W" << width << " H" << height << " F" << fps << ":1 Ip A1:1 C444\n";
    std::vector<unsigned char> yuv(3 * plane);
    for (uint32_t t = 0; t < frames; ++t) {
        char frameName[64];
        std::snprintf(frameName, sizeof(frameName), "frame_%03u.ppm", t);
        std::ofstream ppm(directory / frameName, std::ios::binary);
        ppm << "P6\n" << width << " " << height << "\n255\n";
        for (size_t i = 0; i < plane; ++i) {
            const size_t offset = size_t(t) * plane + i;
            const unsigned char rgb[3] = {byte(video[offset]), byte(video[channelStride + offset]),
                                          byte(video[2 * channelStride + offset])};
            ppm.write(reinterpret_cast<const char*>(rgb), 3);
            // BT.601 full-range conversion.
            const float r = rgb[0], g = rgb[1], b = rgb[2];
            yuv[i] = static_cast<unsigned char>(std::clamp(0.299f * r + 0.587f * g + 0.114f * b + 0.5f, 0.0f, 255.0f));
            yuv[plane + i] = static_cast<unsigned char>(
                std::clamp(-0.168736f * r - 0.331264f * g + 0.5f * b + 128.5f, 0.0f, 255.0f));
            yuv[2 * plane + i] =
                static_cast<unsigned char>(std::clamp(0.5f * r - 0.418688f * g - 0.081312f * b + 128.5f, 0.0f, 255.0f));
        }
        y4m << "FRAME\n";
        y4m.write(reinterpret_cast<const char*>(yuv.data()), yuv.size());
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parseOptions(argc, argv);
        const Config config(options.modelDirectory / "config.txt");
        klartraum::HeadlessFrontend frontend;
        auto& context = frontend.getKlartraumEngine().getVulkanContext();
        const auto& directory = options.modelDirectory;
        const LatentShape shape{config.count("latent_shape", 0), config.count("latent_shape", 1),
                                config.count("latent_shape", 2), config.count("latent_shape", 3),
                                config.count("patch")};
        const uint32_t size = config.count("size");
        const uint32_t frames = config.count("num_frames");
        const size_t frameLatentElements = size_t(shape.height) * shape.width;
        const auto totalStarted = std::chrono::steady_clock::now();

        const uint32_t decoderTile = config.countOr("decoder_tile", 0);
        const uint32_t decoderStride = config.countOr("decoder_stride", 0);
        const bool decoderChunked = config.countOr("decoder_chunked", 0) != 0;
        const uint32_t decoderCaches = config.countOr("decoder_caches", 0);
        if (options.stage == "vae" && decoderChunked) {
            const auto input = readTensor<float>(directory / "decoder_reference_input_f32.bin", shape.elements());
            const auto video = decodeChunked(context, options, shape, input, decoderCaches, size, size);
            requireClose(
                "Decoded video",
                compare(video, readTensor<float>(directory / "decoder_reference_output_f32.bin", video.size())), 1e-3f);
            return 0;
        }
        if (options.stage == "vae") {
            // Decoder alone on the exported decoder fixture, at the decoder's own (tile) size.
            const uint32_t h = decoderTile ? decoderTile : shape.height, w = decoderTile ? decoderTile : shape.width;
            const auto input = readTensor<float>(directory / "decoder_reference_input_f32.bin",
                                                 size_t(shape.channels) * shape.frames * h * w);
            const auto video = runOnce(context, options, "vae_decoder.onnx", "latents",
                                       {1, shape.channels, shape.frames, h, w}, input, "video", "VAE decoder");
            requireClose(
                "Decoded video",
                compare(video, readTensor<float>(directory / "decoder_reference_output_f32.bin", video.size())), 1e-3f);
            return 0;
        }

        // 1. Conditioning frame: Wan VAE encode of the preprocessed image.
        const auto exportedImage = readTensor<float>(directory / "image_f32.bin", size_t(3) * size * size);
        std::vector<float> image = exportedImage;
        if (!options.image.empty()) {
            image = klartraum::preprocessConditioningImage(klartraum::readPpm(options.image), size, size);
            requireClose("Conditioning image", compare(image, exportedImage), 1e-5f);
        }
        auto condition = runOnce(context, options, "vae_encoder.onnx", "image", {1, 3, 1, size, size}, image, "latent",
                                 "VAE encoder");
        requireClose("Condition latent",
                     compare(condition, readTensor<float>(directory / "condition_latent_f32.bin", condition.size())),
                     1e-3f);

        // Frame 0 is the clean condition; the other latent frames start as the exported noise.
        std::vector<float> latents = readTensor<float>(directory / "initial_latents_f32.bin", shape.elements());
        for (uint32_t c = 0; c < shape.channels; ++c) {
            std::copy_n(condition.begin() + size_t(c) * frameLatentElements, frameLatentElements,
                        latents.begin() + size_t(c) * shape.frames * frameLatentElements);
        }

        // 2. Text tower, once for both guidance prompts.
        auto textKeyValues = runTextTower(context, config, options);
        if (options.stage == "text")
            return 0;

        // 3. Denoising: both guidance branches per submit, guidance and UniPC on the host.
        klartraum::UniPCFlowScheduler scheduler(config.count("steps"), config.number("flow_shift"));
        const auto& timesteps = scheduler.timesteps();
        const auto referenceTimesteps = readTensor<float>(directory / "timesteps_f32.bin", timesteps.size());
        if (timesteps != referenceTimesteps)
            throw std::runtime_error("Scheduler timesteps differ from the export");
        const float guidance = static_cast<float>(config.number("guidance_scale"));
        const auto referenceLatents = options.check ? readTensor<float>(directory / "reference_step_latents_f32.bin",
                                                                        timesteps.size() * shape.elements())
                                                    : std::vector<float>{};
        const size_t steps = std::min(timesteps.size(), options.maxSteps);
        double denoiseSeconds = 0.0;
        {
            Denoiser denoiser(context, config, options, shape, textKeyValues);
            textKeyValues = {};
            std::vector<float> guided(shape.elements());
            for (size_t step = 0; step < steps; ++step) {
                const auto started = std::chrono::steady_clock::now();
                const auto velocity = denoiser.predict(latents, timesteps[step]);
                if (step == 0 && options.check) {
                    requireClose("Step-0 velocity",
                                 compare(velocity, readTensor<float>(directory / "denoiser_step0_output_f32.bin",
                                                                     velocity.size())),
                                 1e-3f);
                }
                const size_t branch = shape.tokens() * shape.tokenWidth();
                const auto conditional = unpatchify(velocity.data(), shape);
                const auto unconditional = unpatchify(velocity.data() + branch, shape);
                for (size_t i = 0; i < guided.size(); ++i) {
                    guided[i] = unconditional[i] + guidance * (conditional[i] - unconditional[i]);
                }
                // The conditioning frame stays clean.
                for (uint32_t c = 0; c < shape.channels; ++c) {
                    std::fill_n(guided.begin() + size_t(c) * shape.frames * frameLatentElements, frameLatentElements,
                                0.0f);
                }
                scheduler.step(guided, latents);
                const double seconds = secondsSince(started);
                denoiseSeconds += seconds;
                std::cout << "Step " << step + 1 << "/" << timesteps.size() << " t=" << timesteps[step] << " in "
                          << seconds << " s";
                if (options.check) {
                    const std::vector<float> reference(referenceLatents.begin() + step * shape.elements(),
                                                       referenceLatents.begin() + (step + 1) * shape.elements());
                    std::cout << ", latent max error vs float32 reference " << compare(latents, reference).maxError;
                }
                std::cout << std::endl;
            }
            if (options.profile && steps > 0)
                printProfiling("Denoiser", denoiser.computeGraph());
        }
        if (options.stage == "step")
            return 0;
        if (steps == timesteps.size() && options.check) {
            requireClose(
                "Final latents",
                compare(latents, readTensor<float>(directory / "reference_final_latents_f32.bin", latents.size())),
                2e-2f);
        }

        // 4. Wan VAE decode: of the whole clip, one latent frame at a time, or in overlapping tiles.
        auto video = decoderChunked ? decodeChunked(context, options, shape, latents, decoderCaches, size, size)
                     : decoderTile  ? decodeTiled(context, options, shape, latents, decoderTile, decoderStride)
                                    : runOnce(context, options, "vae_decoder.onnx", "latents",
                                              {1, shape.channels, shape.frames, shape.height, shape.width}, latents,
                                              "video", "VAE decoder");
        for (auto& value : video)
            value = std::clamp(value, -1.0f, 1.0f);
        if (steps == timesteps.size() && options.check) {
            const auto reference = readTensor<float>(directory / "reference_video_f32.bin", video.size());
            const auto check = compare(video, reference);
            double meanError = 0.0;
            for (size_t i = 0; i < video.size(); ++i)
                meanError += std::abs(video[i] - reference[i]);
            meanError /= double(video.size());
            std::cout << "Video vs float32 reference: max error " << check.maxError << ", mean error " << meanError
                      << std::endl;
            if (!(meanError < 1e-2))
                throw std::runtime_error("Decoded video is outside its reference tolerance");
        }
        // Frame 0 reproduces the conditioning image.
        double frameZeroError = 0.0;
        const size_t plane = size_t(size) * size;
        for (uint32_t c = 0; c < 3; ++c) {
            for (size_t i = 0; i < plane; ++i) {
                frameZeroError += std::abs(video[size_t(c) * frames * plane + i] - image[size_t(c) * plane + i]);
            }
        }
        std::cout << "Frame 0 mean error vs the conditioning image: " << frameZeroError / (3.0 * plane) << std::endl;
        writeVideo(options.outputDirectory, options.name, video, frames, size, size,
                   static_cast<uint32_t>(config.number("fps")));
        std::cout << "Denoising " << denoiseSeconds << " s (" << denoiseSeconds / std::max<size_t>(steps, 1)
                  << " s/step), total " << secondsSince(totalStarted) << " s" << std::endl;
        std::cout << "Wrote " << (options.outputDirectory / (options.name + "_klartraum.y4m")) << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Cosmos3 example failed: " << error.what() << std::endl;
        return 1;
    }
}
