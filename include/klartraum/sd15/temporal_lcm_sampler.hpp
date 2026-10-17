// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_SD15_TEMPORAL_LCM_SAMPLER_HPP
#define KLARTRAUM_SD15_TEMPORAL_LCM_SAMPLER_HPP

#include <cstdint>
#include <functional>
#include <vector>

#include "klartraum/sd15/lcm_scheduler.hpp"

namespace klartraum {

/**
 * @brief Samples a sequence of frames with an LCM so that consecutive frames stay consistent, cheaply.
 *
 * Two inexpensive measures, no extra network:
 * - Shared noise: every frame uses the same noise in every step, so what
 *   changes between frames is mostly the conditioning (e.g. the pose map).
 * - Keyframe img2img: before the first frame, a keyframe is sampled from pure
 *   noise through all steps, conditioned like the first frame. Every frame,
 *   the first included, then starts from
 *   `(1 - anchorWeight) * previous + anchorWeight * keyframe` (previous: the
 *   final latents of the frame before; the keyframe for the first frame),
 *   noised to the timestep of `startStep`, and runs only the remaining steps.
 *   With anchorWeight 1 all frames are img2img variations of the same keyframe
 *   under their own conditioning: appearance and background stay put, and
 *   nothing accumulates from frame to frame. Smaller weights follow the
 *   previous frame more closely but let each pass's small shift in contrast and
 *   colour accumulate. Running only the last steps also makes frames cheaper.
 *
 * With `independent` set, every frame runs all steps from the shared noise;
 * that is the baseline the temporal measures are compared against.
 */
class TemporalLcmSampler {
public:
    struct Settings {
        size_t startStep = 1;
        float anchorWeight = 1.0f;
        bool independent = false;
    };

    /// Predicts the noise in `latents` at `timestep` for the frame being sampled.
    using NoisePredictor = std::function<std::vector<float>(const std::vector<float>& latents, int64_t timestep)>;

    /**
     * @param noise noise per scheduler step, each as many values as a latent: entry 0 is the initial latent of
     *        the keyframe, entry k the noise that brings a latent to the timestep of step k (k > 0)
     */
    TemporalLcmSampler(LcmScheduler scheduler, std::vector<std::vector<float>> noise, Settings settings);

    /// Samples the next frame and returns its final latents.
    std::vector<float> sampleFrame(const NoisePredictor& predict);

    /// Noise predictions the next frame will take (the first frame includes the keyframe).
    size_t evaluationsOfNextFrame() const;
    size_t framesSampled() const { return frameCount; }
    /// The keyframe's latents; empty before the first frame or with `independent`.
    const std::vector<float>& getKeyframe() const { return keyframe; }

private:
    // Runs the scheduler steps from `first` on and returns the final latents.
    std::vector<float> run(size_t first, std::vector<float> latents, const NoisePredictor& predict) const;

    LcmScheduler scheduler;
    std::vector<std::vector<float>> noise;
    Settings settings;
    size_t frameCount = 0;
    std::vector<float> keyframe;
    std::vector<float> previousLatents;
};

} // namespace klartraum

#endif // KLARTRAUM_SD15_TEMPORAL_LCM_SAMPLER_HPP
