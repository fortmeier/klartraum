// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include <stdexcept>

#include "klartraum/sd15/temporal_lcm_sampler.hpp"

namespace klartraum {

TemporalLcmSampler::TemporalLcmSampler(LcmScheduler scheduler, std::vector<std::vector<float>> noise, Settings settings)
    : scheduler(std::move(scheduler)),
      noise(std::move(noise)),
      settings(settings) {
    if (this->noise.size() != this->scheduler.size()) {
        throw std::runtime_error("TemporalLcmSampler: needs one noise latent per scheduler step");
    }
    if (settings.startStep == 0 || settings.startStep >= this->scheduler.size()) {
        throw std::runtime_error("TemporalLcmSampler: startStep must skip at least one step and leave at least one");
    }
    for (const auto& latent : this->noise) {
        if (latent.size() != this->noise.front().size()) {
            throw std::runtime_error("TemporalLcmSampler: noise latents differ in size");
        }
    }
}

size_t TemporalLcmSampler::evaluationsOfNextFrame() const {
    const size_t all = scheduler.size(), remaining = scheduler.size() - settings.startStep;
    if (settings.independent) {
        return all;
    }
    return frameCount == 0 ? all + remaining : remaining;
}

std::vector<float> TemporalLcmSampler::run(size_t first, std::vector<float> latents,
                                           const NoisePredictor& predict) const {
    static const std::vector<float> none;
    std::vector<float> denoised;
    for (size_t step = first; step < scheduler.size(); ++step) {
        const auto prediction = predict(latents, scheduler.getSteps()[step].timestep);
        const bool last = step + 1 == scheduler.size();
        scheduler.step(step, latents, prediction, last ? none : noise[step + 1], denoised);
    }
    return latents;
}

std::vector<float> TemporalLcmSampler::sampleFrame(const NoisePredictor& predict) {
    ++frameCount;
    if (settings.independent) {
        return run(0, noise[0], predict);
    }
    if (keyframe.empty()) {
        keyframe = run(0, noise[0], predict);
        previousLatents = keyframe;
    }
    std::vector<float> start(keyframe.size());
    for (size_t i = 0; i < start.size(); ++i) {
        start[i] = (1.0f - settings.anchorWeight) * previousLatents[i] + settings.anchorWeight * keyframe[i];
    }
    // In a full run from noise, the latents entering step startStep were noised with noise[startStep].
    previousLatents =
        run(settings.startStep, scheduler.addNoise(settings.startStep, start, noise[settings.startStep]), predict);
    return previousLatents;
}

} // namespace klartraum
