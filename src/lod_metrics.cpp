// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/lod_metrics.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace klartraum {

std::vector<float> imageLuminance(const std::vector<uint8_t>& bgra) {
    std::vector<float> l(bgra.size() / 4);
    for (size_t i = 0; i < l.size(); ++i) {
        l[i] = (0.0722f * bgra[4 * i] + 0.7152f * bgra[4 * i + 1] + 0.2126f * bgra[4 * i + 2]) / 255.0f;
    }
    return l;
}

float poppingResidual(const std::vector<float>& previous, const std::vector<float>& current,
                      const std::vector<float>& next, uint32_t width, uint32_t height, PoppingConfig config) {
    const size_t pixels = size_t(width) * height;
    if (previous.size() != pixels || current.size() != pixels || next.size() != pixels || config.block == 0) {
        throw std::invalid_argument("poppingResidual: frames must have width x height pixels");
    }
    const uint32_t b = config.block;
    const double area = double(b) * b;
    double sum = 0.0;
    size_t blocks = 0;
    for (uint32_t by = 0; by + b <= height; by += b) {
        for (uint32_t bx = 0; bx + b <= width; bx += b) {
            double lightPrevious = 0.0, lightCurrent = 0.0, lightNext = 0.0;
            for (uint32_t y = by; y < by + b; ++y) {
                for (uint32_t x = bx; x < bx + b; ++x) {
                    const size_t i = size_t(y) * width + x;
                    lightPrevious += previous[i];
                    lightCurrent += current[i];
                    lightNext += next[i];
                }
            }
            if (std::max(lightPrevious, std::max(lightCurrent, lightNext)) / area > config.minLight) {
                sum += std::abs(lightCurrent - 0.5 * (lightPrevious + lightNext)) / area;
                ++blocks;
            }
        }
    }
    return blocks > 0 ? float(sum / double(blocks)) : 0.0f;
}

PoppingMeter::PoppingMeter(uint32_t width, uint32_t height, PoppingConfig config)
    : width(width),
      height(height),
      config(config) {}

std::optional<float> PoppingMeter::push(std::vector<float> frame) {
    std::optional<float> residual;
    if (frames.size() == 2) {
        residual = poppingResidual(frames[0], frames[1], frame, width, height, config);
        frames.erase(frames.begin());
    }
    frames.push_back(std::move(frame));
    return residual;
}

void PoppingMeter::reset() { frames.clear(); }

SeriesSummary summarizeSeries(const std::vector<float>& values) {
    SeriesSummary s;
    if (values.empty()) {
        return s;
    }
    double sum = 0.0;
    for (size_t i = 0; i < values.size(); ++i) {
        sum += values[i];
        if (values[i] > s.max) {
            s.max = values[i];
            s.worst = i;
        }
    }
    s.mean = float(sum / double(values.size()));
    std::vector<float> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    const size_t n = sorted.size();
    s.median = n % 2 == 1 ? sorted[n / 2] : 0.5f * (sorted[n / 2 - 1] + sorted[n / 2]);
    s.spike = s.median > 0.0f ? s.max / s.median : 0.0f;
    return s;
}

float fidelityError(const std::vector<uint8_t>& image, const std::vector<uint8_t>& reference, uint32_t width,
                    uint32_t height, FidelityConfig config) {
    const size_t bytes = size_t(width) * height * 4;
    if (image.size() != bytes || reference.size() != bytes || config.block == 0) {
        throw std::invalid_argument("fidelityError: images must have width x height BGRA pixels");
    }
    const uint32_t b = config.block;
    const double scale = 1.0 / (double(b) * b * 255.0);
    double sum = 0.0;
    size_t blocks = 0;
    for (uint32_t by = 0; by + b <= height; by += b) {
        for (uint32_t bx = 0; bx + b <= width; bx += b) {
            double a[3] = {0.0, 0.0, 0.0}, r[3] = {0.0, 0.0, 0.0};
            for (uint32_t y = by; y < by + b; ++y) {
                for (uint32_t x = bx; x < bx + b; ++x) {
                    const size_t i = (size_t(y) * width + x) * 4;
                    for (int c = 0; c < 3; ++c) {
                        a[c] += image[i + c];
                        r[c] += reference[i + c];
                    }
                }
            }
            const double lightImage = (a[0] + a[1] + a[2]) * scale / 3.0;
            const double lightReference = (r[0] + r[1] + r[2]) * scale / 3.0;
            if (std::max(lightImage, lightReference) > config.minLight) {
                sum += (std::abs(a[0] - r[0]) + std::abs(a[1] - r[1]) + std::abs(a[2] - r[2])) * scale / 3.0;
                ++blocks;
            }
        }
    }
    return blocks > 0 ? float(sum / double(blocks)) : 0.0f;
}

float detailError(const std::vector<uint8_t>& image, const std::vector<uint8_t>& reference, uint32_t width,
                  uint32_t height, DetailConfig config) {
    const size_t bytes = size_t(width) * height * 4;
    if (image.size() != bytes || reference.size() != bytes || config.block == 0) {
        throw std::invalid_argument("detailError: images must have width x height BGRA pixels");
    }
    const std::vector<float> a = imageLuminance(image), b = imageLuminance(reference);
    // The 4-neighbour Laplacian at (x, y), edges clamped.
    auto laplacian = [&](const std::vector<float>& l, uint32_t x, uint32_t y) {
        const uint32_t x0 = x > 0 ? x - 1 : x, x1 = x + 1 < width ? x + 1 : x;
        const uint32_t y0 = y > 0 ? y - 1 : y, y1 = y + 1 < height ? y + 1 : y;
        const size_t i = size_t(y) * width + x;
        return 4.0f * l[i] - l[size_t(y) * width + x0] - l[size_t(y) * width + x1] - l[size_t(y0) * width + x] -
               l[size_t(y1) * width + x];
    };
    const uint32_t blockSize = config.block;
    const double area = double(blockSize) * blockSize;
    double sum = 0.0;
    size_t blocks = 0;
    for (uint32_t by = 0; by + blockSize <= height; by += blockSize) {
        for (uint32_t bx = 0; bx + blockSize <= width; bx += blockSize) {
            double energyA = 0.0, energyB = 0.0, lightA = 0.0, lightB = 0.0;
            for (uint32_t y = by; y < by + blockSize; ++y) {
                for (uint32_t x = bx; x < bx + blockSize; ++x) {
                    const float la = laplacian(a, x, y), lb = laplacian(b, x, y);
                    energyA += double(la) * la;
                    energyB += double(lb) * lb;
                    lightA += a[size_t(y) * width + x];
                    lightB += b[size_t(y) * width + x];
                }
            }
            if (std::max(lightA, lightB) / area > config.minLight) {
                sum += std::abs(std::log2((energyA / area + config.floor) / (energyB / area + config.floor)));
                ++blocks;
            }
        }
    }
    return blocks > 0 ? float(sum / double(blocks)) : 0.0f;
}

} // namespace klartraum
