// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/cosmos3/conditioning_image.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>

namespace klartraum {

namespace {

/** Reads the next whitespace-separated header token, skipping '#' comments. */
std::string headerToken(std::istream& stream) {
    std::string token;
    while (stream >> token) {
        if (token[0] != '#')
            return token;
        std::string rest;
        std::getline(stream, rest);
    }
    return {};
}

/** Taps and normalized weights of one output sample. */
struct Taps {
    int64_t first = 0;
    std::vector<float> weights;
};

/**
 * Antialiased bilinear weights as torch computes them (upsample_bilinear2d_aa,
 * align_corners=false): a triangle filter widened by the scale factor when
 * downscaling, truncated at the borders and renormalized. Computed in float,
 * like torch for float tensors.
 */
std::vector<Taps> antialiasedBilinearTaps(int64_t inputSize, int64_t outputSize) {
    const float scale = float(inputSize) / float(outputSize);
    const float support = scale >= 1.0f ? scale : 1.0f;
    const float invScale = scale >= 1.0f ? 1.0f / scale : 1.0f;
    std::vector<Taps> taps(static_cast<size_t>(outputSize));
    for (int64_t i = 0; i < outputSize; ++i) {
        const float center = scale * (float(i) + 0.5f);
        const int64_t first = std::max<int64_t>(int64_t(center - support + 0.5f), 0);
        const int64_t count = std::min<int64_t>(int64_t(center + support + 0.5f), inputSize) - first;
        auto& tap = taps[size_t(i)];
        tap.first = first;
        tap.weights.resize(size_t(std::max<int64_t>(count, 0)));
        float total = 0.0f;
        for (int64_t j = 0; j < count; ++j) {
            const float w = std::max(0.0f, 1.0f - std::abs((float(j + first) - center + 0.5f) * invScale));
            tap.weights[size_t(j)] = w;
            total += w;
        }
        if (total != 0.0f) {
            for (auto& w : tap.weights)
                w /= total;
        }
    }
    return taps;
}

} // namespace

RgbImage readPpm(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error("Cannot open " + path.string());
    if (headerToken(file) != "P6")
        throw std::runtime_error(path.string() + " is not a binary (P6) PPM");
    RgbImage image;
    int maxValue = 0;
    try {
        image.width = uint32_t(std::stoul(headerToken(file)));
        image.height = uint32_t(std::stoul(headerToken(file)));
        maxValue = std::stoi(headerToken(file));
    } catch (const std::exception&) {
        throw std::runtime_error(path.string() + " has a malformed PPM header");
    }
    if (maxValue != 255)
        throw std::runtime_error(path.string() + " is not an 8-bit PPM");
    file.get(); // the single whitespace byte that ends the header
    image.pixels.resize(size_t(image.width) * image.height * 3);
    file.read(reinterpret_cast<char*>(image.pixels.data()), std::streamsize(image.pixels.size()));
    if (!file)
        throw std::runtime_error(path.string() + " is truncated");
    return image;
}

std::vector<float> preprocessConditioningImage(const RgbImage& image, uint32_t width, uint32_t height) {
    if (image.width == 0 || image.height == 0 || image.pixels.size() != size_t(image.width) * image.height * 3) {
        throw std::runtime_error("preprocessConditioningImage: invalid image");
    }
    // Scale to cover the target, sizes rounded up, then crop the centre.
    const double scale = std::max(double(width) / image.width, double(height) / image.height);
    const auto resizedWidth = int64_t(std::ceil(scale * image.width));
    const auto resizedHeight = int64_t(std::ceil(scale * image.height));
    // Python's round() rounds halves to even, as std::nearbyint does by default.
    const auto cropLeft = int64_t(std::nearbyint(double(resizedWidth - width) / 2.0));
    const auto cropTop = int64_t(std::nearbyint(double(resizedHeight - height) / 2.0));

    const auto columns = antialiasedBilinearTaps(image.width, resizedWidth);
    const auto rows = antialiasedBilinearTaps(image.height, resizedHeight);

    // Horizontal pass over all source rows, for the cropped columns only.
    std::vector<float> horizontal(size_t(image.height) * width * 3);
    for (uint32_t y = 0; y < image.height; ++y) {
        const uint8_t* row = image.pixels.data() + size_t(y) * image.width * 3;
        for (uint32_t x = 0; x < width; ++x) {
            const auto& tap = columns[size_t(cropLeft + x)];
            for (uint32_t c = 0; c < 3; ++c) {
                float sum = 0.0f;
                for (size_t j = 0; j < tap.weights.size(); ++j) {
                    sum += tap.weights[j] * float(row[(size_t(tap.first) + j) * 3 + c]);
                }
                horizontal[(size_t(y) * width + x) * 3 + c] = sum;
            }
        }
    }

    // Vertical pass for the cropped rows, rounded to levels and mapped to [-1, 1].
    std::vector<float> tensor(size_t(3) * height * width);
    for (uint32_t y = 0; y < height; ++y) {
        const auto& tap = rows[size_t(cropTop + y)];
        for (uint32_t x = 0; x < width; ++x) {
            for (uint32_t c = 0; c < 3; ++c) {
                float sum = 0.0f;
                for (size_t j = 0; j < tap.weights.size(); ++j) {
                    sum += tap.weights[j] * horizontal[((size_t(tap.first) + j) * width + x) * 3 + c];
                }
                const float pixel = std::clamp(std::nearbyint(sum), 0.0f, 255.0f);
                tensor[(size_t(c) * height + y) * width + x] = pixel / 127.5f - 1.0f;
            }
        }
    }
    return tensor;
}

} // namespace klartraum
