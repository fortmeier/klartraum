// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/neural/photo_enhancer.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include <glm/glm.hpp>

#include "klartraum/neural/hash_texture.hpp"
#include "klartraum/onnx/onnx_network.hpp"
#include "klartraum/tree/tree_model.hpp"

namespace klartraum::neural {

namespace {

constexpr uint32_t kClasses = 7, kChannels = 3 + 1 + kClasses;

float value16(const std::vector<uint8_t>& bgra, size_t p) {
    return float(bgra[4 * p + 2] * 256u + bgra[4 * p + 1]) / 65535.0f;
}

// One pass of PIL's resampling (Image.resize with BICUBIC or BILINEAR): output sample i of `outSize` blends the inputs
// around (i + 0.5) * scale with the filter stretched by the scale, the weights normalized.
struct Taps {
    std::vector<uint32_t> first;
    std::vector<std::vector<float>> weights;
};

Taps resamplingTaps(uint32_t inSize, uint32_t outSize, bool bicubic) {
    const auto filter = [bicubic](float x) {
        x = std::abs(x);
        if (!bicubic)
            return x < 1.0f ? 1.0f - x : 0.0f;
        constexpr float a = -0.5f;
        if (x < 1.0f)
            return ((a + 2.0f) * x - (a + 3.0f)) * x * x + 1.0f;
        if (x < 2.0f)
            return (((x - 5.0f) * x + 8.0f) * x - 4.0f) * a;
        return 0.0f;
    };
    const float scale = float(inSize) / float(outSize), filterScale = std::max(scale, 1.0f);
    const float support = (bicubic ? 2.0f : 1.0f) * filterScale;
    Taps taps;
    for (uint32_t i = 0; i < outSize; ++i) {
        const float centre = (float(i) + 0.5f) * scale;
        const int lo = std::max(int(centre - support + 0.5f), 0);
        const int hi = std::min(int(centre + support + 0.5f), int(inSize));
        std::vector<float> w;
        float sum = 0.0f;
        for (int x = lo; x < hi; ++x) {
            w.push_back(filter((float(x) - centre + 0.5f) / filterScale));
            sum += w.back();
        }
        for (float& v : w)
            v /= sum;
        taps.first.push_back(uint32_t(lo));
        taps.weights.push_back(std::move(w));
    }
    return taps;
}

// How a frame maps to the sample: scaled to the sample width, then cropped to its height around the middle.
struct Resampling {
    Resampling(uint32_t renderWidth, uint32_t renderHeight, const PhotoEnhancerSample& sample)
        : renderWidth(renderWidth),
          renderHeight(renderHeight),
          width(sample.width),
          height(sample.height),
          scaledHeight(uint32_t(std::lround(double(renderHeight) * sample.width / renderWidth))),
          top((scaledHeight - sample.height) / 2) {
        if (scaledHeight < sample.height)
            throw std::invalid_argument("photoEnhancerInputs: the frame is too flat for the sample");
        for (int bicubic = 0; bicubic < 2; ++bicubic) {
            horizontal[bicubic] = resamplingTaps(renderWidth, width, bicubic);
            vertical[bicubic] = resamplingTaps(renderHeight, scaledHeight, bicubic);
        }
    }

    // A [renderHeight][renderWidth] plane at the sample size; with `bytes`, rounded to whole 8-bit levels after each
    // pass as PIL does for 8-bit images.
    std::vector<float> apply(const std::vector<float>& plane, bool bicubic, bool bytes) const {
        const auto level = [bytes](float v) { return bytes ? std::clamp(std::round(v), 0.0f, 255.0f) : v; };
        const Taps& h = horizontal[bicubic];
        const Taps& v = vertical[bicubic];
        std::vector<float> rows(size_t(renderHeight) * width), out(size_t(height) * width);
        for (uint32_t y = 0; y < renderHeight; ++y)
            for (uint32_t x = 0; x < width; ++x) {
                float sum = 0.0f;
                for (size_t k = 0; k < h.weights[x].size(); ++k)
                    sum += h.weights[x][k] * plane[size_t(y) * renderWidth + h.first[x] + k];
                rows[size_t(y) * width + x] = level(sum);
            }
        for (uint32_t y = 0; y < height; ++y)
            for (uint32_t x = 0; x < width; ++x) {
                const uint32_t row = y + top;
                float sum = 0.0f;
                for (size_t k = 0; k < v.weights[row].size(); ++k)
                    sum += v.weights[row][k] * rows[(v.first[row] + k) * width + x];
                out[size_t(y) * width + x] = level(sum);
            }
        return out;
    }

    // The frame pixel PIL's NEAREST resampling takes for sample pixel p.
    size_t nearest(size_t p) const {
        const uint32_t x = uint32_t(p % width), y = uint32_t(p / width) + top;
        const uint32_t sx = uint32_t((float(x) + 0.5f) * float(renderWidth) / float(width));
        const uint32_t sy = uint32_t((float(y) + 0.5f) * float(renderHeight) / float(scaledHeight));
        return size_t(sy) * renderWidth + sx;
    }

    uint32_t renderWidth, renderHeight, width, height, scaledHeight, top;
    Taps horizontal[2], vertical[2];
};

std::vector<float> readFloats(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
        throw std::runtime_error("PhotoEnhancer: cannot read " + path);
    std::vector<float> data(size_t(file.tellg()) / sizeof(float));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(data.data()), std::streamsize(data.size() * sizeof(float)));
    return data;
}

} // namespace

PhotoEnhancerInputs photoEnhancerInputs(const GBufferFrame& frame, const PhotoEnhancerSample& sample) {
    const Resampling resampling(frame.width, frame.height, sample);
    const size_t renderPixels = size_t(frame.width) * frame.height, pixels = size_t(sample.width) * sample.height;
    // Classes: the nearest class colour; the background (black) is the only uncovered class.
    std::vector<uint8_t> classes(renderPixels);
    for (size_t p = 0; p < renderPixels; ++p) {
        const glm::vec3 c =
            glm::vec3(frame.classes[4 * p + 2], frame.classes[4 * p + 1], frame.classes[4 * p]) / 255.0f;
        uint32_t best = 0;
        for (uint32_t i = 1; i < kClasses; ++i)
            if (glm::length(tree::treeClassColor(i) - c) < glm::length(tree::treeClassColor(best) - c))
                best = i;
        classes[p] = uint8_t(best);
    }
    PhotoEnhancerInputs inputs;
    inputs.channels.assign(size_t(kChannels) * pixels, 0.0f);
    inputs.coordinates.resize(3 * pixels);
    inputs.classes.resize(pixels);
    std::vector<float> plane(renderPixels), inverse(renderPixels, 0.0f), coveredInverse;
    for (int c = 0; c < 3; ++c) {
        for (size_t p = 0; p < renderPixels; ++p)
            plane[p] = frame.colour[4 * p + 2 - c];
        const auto scaled = resampling.apply(plane, true, true);
        for (size_t p = 0; p < pixels; ++p)
            inputs.channels[c * pixels + p] = scaled[p] / 255.0f;
    }
    for (size_t p = 0; p < renderPixels; ++p)
        if (classes[p] != 0) {
            inverse[p] = 1.0f / std::max(value16(frame.depth16, p) * frame.depthFar, 1e-9f);
            coveredInverse.push_back(inverse[p]);
        }
    std::fill(plane.begin(), plane.end(), 0.0f);
    if (!coveredInverse.empty()) {
        // NumPy's percentile: linear between the two nearest ranks.
        const auto percentile = [&](double q) {
            const double position = q * double(coveredInverse.size() - 1);
            const size_t below = size_t(position);
            std::nth_element(coveredInverse.begin(), coveredInverse.begin() + below, coveredInverse.end());
            const float low = coveredInverse[below];
            if (below + 1 >= coveredInverse.size())
                return low;
            const float high = *std::min_element(coveredInverse.begin() + below + 1, coveredInverse.end());
            return low + float(position - double(below)) * (high - low);
        };
        const float a = percentile(0.02), b = percentile(sample.depthHigh);
        for (size_t p = 0; p < renderPixels; ++p)
            if (classes[p] != 0)
                plane[p] = std::clamp((inverse[p] - a) / std::max(b - a, 1e-12f), 0.0f, 1.0f);
    }
    const auto depth = resampling.apply(plane, false, false);
    std::copy(depth.begin(), depth.end(), inputs.channels.begin() + 3 * pixels);
    for (size_t p = 0; p < pixels; ++p) {
        const size_t q = resampling.nearest(p);
        inputs.channels[(4 + classes[q]) * pixels + p] = 1.0f;
        inputs.classes[p] = float(classes[q]);
        inputs.coordinates[p] = value16(frame.u, q);
        inputs.coordinates[pixels + p] = value16(frame.v, q);
        inputs.coordinates[2 * pixels + p] = float(frame.u[4 * q] * 256u + frame.v[4 * q]) / 65535.0f;
    }
    return inputs;
}

PhotoEnhancer::PhotoEnhancer(VulkanContext& vulkanContext, const std::string& stem, uint32_t width, uint32_t height)
    : vulkanContext(vulkanContext),
      width(width),
      height(height) {
    HashTextureShape shape = HashTextureShape::fromFile(stem + "_texture.txt");
    if (shape.width != width || shape.height != height)
        throw std::invalid_argument("PhotoEnhancer: " + stem + " was exported for another image size");
    shape.outputOffset = kChannels;
    const auto tableValues = readFloats(stem + "_texture_f32.bin");
    coordinates = vulkanContext.create<TensorElement<float>>(std::vector<uint32_t>{3, height, width});
    classes = vulkanContext.create<TensorElement<float>>(std::vector<uint32_t>{height, width});
    table = vulkanContext.create<TensorElement<float>>(std::vector<uint32_t>{uint32_t(tableValues.size())});
    input = vulkanContext.create<TensorElement<float>>(
        std::vector<uint32_t>{1, kChannels + shape.levels * shape.features, height, width});
    auto texture = hashTexture(vulkanContext, shape);
    texture->setInput(coordinates, 0);
    texture->setInput(classes, 1);
    texture->setInput(table, 2);
    texture->setInput(input, 3);
    textureGraph = std::make_unique<ComputeGraph>(vulkanContext, 1);
    textureGraph->compileFrom(texture);

    network = vulkanContext.create<OnnxNetwork>(stem + ".onnx");
    network->retainTensor("image");
    network->setInputTensor("input", input, -1);
    if (std::filesystem::exists(stem + "_season_f32.bin")) {
        season = vulkanContext.create<TensorElement<float>>(std::vector<uint32_t>{1, 2});
        network->setInputTensor("season", season, -1);
    }
    if (std::filesystem::exists(stem + "_appearance_input_f32.bin")) {
        codeSize = uint32_t(readFloats(stem + "_appearance_input_f32.bin").size());
        codes = readFloats(stem + "_appearance_codes_f32.bin");
        std::ifstream ids(stem + "_appearance_ids.txt");
        for (std::string id; ids >> id;)
            codeIds.push_back(id);
        appearance = vulkanContext.create<TensorElement<float>>(std::vector<uint32_t>{1, codeSize});
        network->setInputTensor("appearance", appearance, -1);
    }
    networkGraph = std::make_unique<ComputeGraph>(vulkanContext, 1);
    networkGraph->compileFrom(network);
    output = std::dynamic_pointer_cast<TensorElement<float>>(network->getOutputElement("image"));
    table->getDataBuffer(0).memcopyFrom(tableValues);
    if (season)
        setSeason(2.0f);
    if (appearance) {
        std::vector<float> mean(codeSize, 0.0f);
        const size_t images = codes.size() / codeSize;
        for (size_t i = 0; i < codes.size(); ++i)
            mean[i % codeSize] += codes[i] / float(std::max<size_t>(images, 1));
        setAppearance(mean);
    }
}

PhotoEnhancer::~PhotoEnhancer() = default;

void PhotoEnhancer::setSeason(float value) {
    if (!season)
        throw std::logic_error("PhotoEnhancer: the generator is not conditioned on the season");
    const float angle = value * 3.14159265f / 2.0f;
    season->getDataBuffer(0).memcopyFrom(std::vector<float>{std::sin(angle), std::cos(angle)});
}

void PhotoEnhancer::setAppearance(const std::vector<float>& code) {
    if (!appearance || code.size() != codeSize)
        throw std::invalid_argument("PhotoEnhancer: the generator takes no appearance code of that size");
    appearance->getDataBuffer(0).memcopyFrom(code);
}

std::vector<float> PhotoEnhancer::run(const PhotoEnhancerInputs& inputs) {
    const size_t pixels = size_t(width) * height;
    if (inputs.channels.size() != kChannels * pixels || inputs.coordinates.size() != 3 * pixels ||
        inputs.classes.size() != pixels)
        throw std::invalid_argument("PhotoEnhancer: inputs of another size");
    input->getDataBuffer(0).memcopyFrom(inputs.channels.data(), inputs.channels.size());
    coordinates->getDataBuffer(0).memcopyFrom(inputs.coordinates);
    classes->getDataBuffer(0).memcopyFrom(inputs.classes);
    textureGraph->submitAndWait(vulkanContext.getGraphicsQueue(), 0);
    networkGraph->submitAndWait(vulkanContext.getGraphicsQueue(), 0);
    std::vector<float> image(output->getDataElementCount());
    output->getDataBuffer(0).memcopyTo(image);
    return image;
}

} // namespace klartraum::neural
