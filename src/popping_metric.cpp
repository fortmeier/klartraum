// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/popping_metric.hpp"

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

PoppingSummary summarizePopping(const std::vector<float>& residuals) {
    PoppingSummary s;
    if (residuals.empty()) {
        return s;
    }
    double sum = 0.0;
    for (size_t i = 0; i < residuals.size(); ++i) {
        sum += residuals[i];
        if (residuals[i] > s.max) {
            s.max = residuals[i];
            s.worst = i;
        }
    }
    s.mean = float(sum / double(residuals.size()));
    std::vector<float> sorted = residuals;
    std::sort(sorted.begin(), sorted.end());
    const size_t n = sorted.size();
    s.median = n % 2 == 1 ? sorted[n / 2] : 0.5f * (sorted[n / 2 - 1] + sorted[n / 2]);
    s.spike = s.median > 0.0f ? s.max / s.median : 0.0f;
    return s;
}

} // namespace klartraum
