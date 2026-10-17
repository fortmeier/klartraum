// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_SD15_LATENT_INPAINTING_HPP
#define KLARTRAUM_SD15_LATENT_INPAINTING_HPP

#include <cstdint>
#include <functional>
#include <vector>

#include "klartraum/sd15/lcm_scheduler.hpp"

namespace klartraum {

/**
 * @brief Downsamples a pixel mask (non-zero: masked) to latent resolution.
 *
 * A latent pixel is masked (1) when any pixel of its `factor` x `factor` block
 * is, then the mask is dilated by `dilation` latent pixels (8-neighbourhood),
 * so the generated region has room around the masked shape. Returns height /
 * factor rows of width / factor values.
 */
std::vector<float> latentMask(const std::vector<uint8_t>& pixels, uint32_t width, uint32_t height, uint32_t factor = 8,
                              uint32_t dilation = 1);

/// What to inpaint: `mask` per latent pixel (1: generate, 0: keep), `known` the clean latent to keep (channel-major).
struct LatentInpainting {
    std::vector<float> mask;
    std::vector<float> known;
};

/**
 * @brief LCM sampling of a latent in which only the masked region is generated, by latent blending.
 *
 * Starts from `noise[0]` in the masked region and from the known latent noised
 * to the first timestep elsewhere. After every step, the unmasked region is
 * replaced by the known latent noised to the next step's timestep (with the
 * same noise the step used), and after the last step by the known latent
 * itself. The model thus sees a consistent scene around the region it fills.
 * Works with any noise-predicting model, no dedicated inpainting UNet.
 *
 * `noise` holds one latent per scheduler step; `channels` latent channels share
 * each mask value.
 */
std::vector<float> sampleInpainted(const LcmScheduler& scheduler, const std::vector<std::vector<float>>& noise,
                                   const LatentInpainting& inpainting,
                                   const std::function<std::vector<float>(const std::vector<float>&, int64_t)>& predict,
                                   size_t channels = 4);

/**
 * @brief A deterministic field of standard normal noise, from which latent-sized windows are cut.
 *
 * Holds `steps` noise planes of `channels` x `height` x `width`. Cutting the
 * window at an offset that follows a moving subject keeps the noise the
 * subject is generated from the same from frame to frame, a cheap way to keep
 * its appearance consistent. Windows wrap around the field's edges.
 */
class NoiseField {
public:
    NoiseField(size_t steps, size_t channels, size_t width, size_t height, uint32_t seed);

    /// Per step, the channel-major window of `width` x `height` whose top-left corner is (left, top).
    std::vector<std::vector<float>> window(int left, int top, size_t width, size_t height) const;

    size_t getWidth() const { return width; }
    size_t getHeight() const { return height; }

private:
    size_t steps, channels, width, height;
    std::vector<float> values;
};

} // namespace klartraum

#endif // KLARTRAUM_SD15_LATENT_INPAINTING_HPP
