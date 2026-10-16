// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_SD15_LCM_SCHEDULER_HPP
#define KLARTRAUM_SD15_LCM_SCHEDULER_HPP

#include <cstdint>
#include <filesystem>
#include <vector>

namespace klartraum {

/**
 * @brief Sampling with a latent consistency model (LCM), e.g. SD1.5 with LCM-LoRA fused in.
 *
 * Follows diffusers' LCMScheduler: each step predicts the clean latent from the
 * noise prediction, applies the consistency boundary condition
 * (c_skip * x + c_out * x0), and, except after the last step, noises the result
 * to the next timestep. The noise is passed in explicitly, so a sequence of
 * frames can share the same noise for temporal consistency.
 *
 * The schedule is exported by scripts/sd15_onnx/export_controlnet.py as
 * `lcm_schedule_f32.bin`: per step the timestep, alpha_t (cumulative product
 * of alphas), alpha of the next timestep, c_skip and c_out.
 */
class LcmScheduler {
public:
    struct Step {
        int64_t timestep;
        float alpha;
        float alphaNext;
        float cSkip;
        float cOut;
    };

    explicit LcmScheduler(std::vector<Step> steps);

    /// Reads `lcm_schedule_f32.bin`.
    static LcmScheduler load(const std::filesystem::path& path);

    const std::vector<Step>& getSteps() const { return steps; }
    size_t size() const { return steps.size(); }

    /**
     * @brief Advances `latents` by step `index` in place and writes the predicted clean latent to `denoised`.
     *
     * `noise` (as many values as `latents`) noises the result to the next
     * timestep; it is ignored after the last step, where the latents become the
     * denoised ones.
     */
    void step(size_t index, std::vector<float>& latents, const std::vector<float>& noisePrediction,
              const std::vector<float>& noise, std::vector<float>& denoised) const;

    /// `sqrt(alpha) * clean + sqrt(1 - alpha) * noise` at the timestep of step `index`, to start sampling there.
    std::vector<float> addNoise(size_t index, const std::vector<float>& clean, const std::vector<float>& noise) const;

private:
    std::vector<Step> steps;
};

} // namespace klartraum

#endif // KLARTRAUM_SD15_LCM_SCHEDULER_HPP
