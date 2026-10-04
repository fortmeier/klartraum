// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_COSMOS3_UNIPC_FLOW_SCHEDULER_HPP
#define KLARTRAUM_COSMOS3_UNIPC_FLOW_SCHEDULER_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

namespace klartraum {

/**
 * @brief UniPC multistep sampler for flow-matching velocity models.
 *
 * Reproduces diffusers' `UniPCMultistepScheduler` as Cosmos3 configures it:
 * flow sigmas with a static shift, data (x0) prediction, the B(h) = e^h - 1
 * ("bh2") variant, lower-order warm-up and final steps, the corrector on every
 * step after the first, and a final sigma of zero. The unshifted schedule is
 * Cosmos3's native one, `linspace(1 - 1/T, 0, steps + 1)` without its last entry.
 */
class UniPCFlowScheduler {
public:
    /**
     * @param steps Number of denoising steps.
     * @param flowShift Static shift `s`: sigma' = s * sigma / (1 + (s - 1) * sigma).
     * @param trainTimesteps Number of training timesteps T.
     * @param solverOrder Multistep order (1 to 3).
     */
    UniPCFlowScheduler(size_t steps, double flowShift, uint32_t trainTimesteps = 1000, uint32_t solverOrder = 2);

    /** @brief steps + 1 noise levels, the last one zero. */
    const std::vector<float>& sigmas() const { return sigmaValues; }

    /** @brief The model timestep of every step, `floor(sigma * T)`. */
    const std::vector<float>& timesteps() const { return timestepValues; }

    /** @brief Index of the next step. */
    size_t stepIndex() const { return currentStep; }

    /**
     * @brief Advances @p sample by one step given the model's velocity at the current step.
     * @throws std::runtime_error If all steps were taken or the sizes differ.
     */
    void step(const std::vector<float>& velocity, std::vector<float>& sample);

private:
    struct Coefficients {
        double sampleScale;         ///< sigma_t / sigma_s0
        double outputScale;         ///< alpha_t * (e^-h - 1)
        double differenceScale;     ///< alpha_t * B(h)
        std::vector<double> rhos;   ///< weights of the scaled output differences
        std::vector<double> ratios; ///< r_k of the earlier outputs
    };

    Coefficients coefficients(size_t target, size_t source, uint32_t order, bool corrector) const;
    double lambda(size_t index) const;

    std::vector<float> sigmaValues;
    std::vector<float> timestepValues;
    uint32_t solverOrder;
    size_t currentStep = 0;
    uint32_t lowerOrderCount = 0;
    uint32_t lastOrder = 0;
    std::vector<std::vector<float>> modelOutputs; ///< x0 predictions, newest last
    std::vector<float> lastSample;
};

} // namespace klartraum

#endif // KLARTRAUM_COSMOS3_UNIPC_FLOW_SCHEDULER_HPP
