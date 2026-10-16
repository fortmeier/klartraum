// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include <cmath>
#include <fstream>
#include <stdexcept>

#include "klartraum/sd15/lcm_scheduler.hpp"

namespace klartraum {

LcmScheduler::LcmScheduler(std::vector<Step> steps)
    : steps(std::move(steps)) {
    if (this->steps.empty()) {
        throw std::runtime_error("LcmScheduler: the schedule has no steps");
    }
}

LcmScheduler LcmScheduler::load(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("LcmScheduler: could not open " + path.string());
    }
    const auto bytes = static_cast<size_t>(input.tellg());
    if (bytes == 0 || bytes % (5 * sizeof(float)) != 0) {
        throw std::runtime_error("LcmScheduler: " + path.string() + " is not a list of 5-float rows");
    }
    std::vector<float> values(bytes / sizeof(float));
    input.seekg(0);
    input.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(bytes));
    std::vector<Step> steps;
    for (size_t row = 0; row < values.size(); row += 5) {
        steps.push_back({static_cast<int64_t>(std::llround(values[row])), values[row + 1], values[row + 2],
                         values[row + 3], values[row + 4]});
    }
    return LcmScheduler(std::move(steps));
}

void LcmScheduler::step(size_t index, std::vector<float>& latents, const std::vector<float>& noisePrediction,
                        const std::vector<float>& noise, std::vector<float>& denoised) const {
    const Step& current = steps.at(index);
    const bool last = index + 1 == steps.size();
    if (noisePrediction.size() != latents.size() || (!last && noise.size() != latents.size())) {
        throw std::runtime_error("LcmScheduler: latents, noise prediction and noise differ in size");
    }
    const float sqrtAlpha = std::sqrt(current.alpha);
    const float sqrtOneMinusAlpha = std::sqrt(1.0f - current.alpha);
    const float sqrtAlphaNext = std::sqrt(current.alphaNext);
    const float sqrtOneMinusAlphaNext = std::sqrt(1.0f - current.alphaNext);
    denoised.resize(latents.size());
    for (size_t i = 0; i < latents.size(); ++i) {
        const float original = (latents[i] - sqrtOneMinusAlpha * noisePrediction[i]) / sqrtAlpha;
        denoised[i] = current.cOut * original + current.cSkip * latents[i];
        latents[i] = last ? denoised[i] : sqrtAlphaNext * denoised[i] + sqrtOneMinusAlphaNext * noise[i];
    }
}

std::vector<float> LcmScheduler::addNoise(size_t index, const std::vector<float>& clean,
                                          const std::vector<float>& noise) const {
    if (clean.size() != noise.size()) {
        throw std::runtime_error("LcmScheduler: clean latents and noise differ in size");
    }
    const float alpha = steps.at(index).alpha;
    const float sqrtAlpha = std::sqrt(alpha), sqrtOneMinusAlpha = std::sqrt(1.0f - alpha);
    std::vector<float> result(clean.size());
    for (size_t i = 0; i < clean.size(); ++i) {
        result[i] = sqrtAlpha * clean[i] + sqrtOneMinusAlpha * noise[i];
    }
    return result;
}

} // namespace klartraum
