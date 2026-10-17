// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <random>
#include <stdexcept>

#include "klartraum/sd15/latent_inpainting.hpp"

namespace klartraum {

std::vector<float> latentMask(const std::vector<uint8_t>& pixels, uint32_t width, uint32_t height, uint32_t factor,
                              uint32_t dilation) {
    if (factor == 0 || width % factor != 0 || height % factor != 0 || pixels.size() != size_t(width) * height) {
        throw std::runtime_error("latentMask: the mask size must be a multiple of the factor");
    }
    const uint32_t latentWidth = width / factor, latentHeight = height / factor;
    std::vector<float> blocks(size_t(latentWidth) * latentHeight, 0.0f);
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            if (pixels[size_t(y) * width + x] != 0) {
                blocks[size_t(y / factor) * latentWidth + x / factor] = 1.0f;
            }
        }
    }
    std::vector<float> result = blocks;
    const int reach = static_cast<int>(dilation);
    for (int y = 0; y < int(latentHeight); ++y) {
        for (int x = 0; x < int(latentWidth); ++x) {
            if (blocks[size_t(y) * latentWidth + size_t(x)] == 0.0f) {
                continue;
            }
            for (int dy = -reach; dy <= reach; ++dy) {
                for (int dx = -reach; dx <= reach; ++dx) {
                    const int nx = x + dx, ny = y + dy;
                    if (nx >= 0 && ny >= 0 && nx < int(latentWidth) && ny < int(latentHeight)) {
                        result[size_t(ny) * latentWidth + size_t(nx)] = 1.0f;
                    }
                }
            }
        }
    }
    return result;
}

std::vector<float> sampleInpainted(const LcmScheduler& scheduler, const std::vector<std::vector<float>>& noise,
                                   const LatentInpainting& inpainting,
                                   const std::function<std::vector<float>(const std::vector<float>&, int64_t)>& predict,
                                   size_t channels) {
    const size_t plane = inpainting.mask.size();
    if (noise.size() != scheduler.size() || inpainting.known.size() != channels * plane) {
        throw std::runtime_error("sampleInpainted: noise, mask and known latent do not fit together");
    }
    for (const auto& latent : noise) {
        if (latent.size() != channels * plane) {
            throw std::runtime_error("sampleInpainted: a noise latent has the wrong size");
        }
    }
    // Replaces the unmasked region of `latents` with `outside`.
    auto keepOutside = [&](std::vector<float>& latents, const std::vector<float>& outside) {
        for (size_t i = 0; i < latents.size(); ++i) {
            const float mask = inpainting.mask[i % plane];
            latents[i] = mask * latents[i] + (1.0f - mask) * outside[i];
        }
    };

    std::vector<float> latents = noise[0];
    keepOutside(latents, scheduler.addNoise(0, inpainting.known, noise[0]));
    std::vector<float> denoised;
    static const std::vector<float> none;
    for (size_t step = 0; step < scheduler.size(); ++step) {
        const auto prediction = predict(latents, scheduler.getSteps()[step].timestep);
        const bool last = step + 1 == scheduler.size();
        scheduler.step(step, latents, prediction, last ? none : noise[step + 1], denoised);
        keepOutside(latents, last ? inpainting.known : scheduler.addNoise(step + 1, inpainting.known, noise[step + 1]));
    }
    return latents;
}

NoiseField::NoiseField(size_t steps, size_t channels, size_t width, size_t height, uint32_t seed)
    : steps(steps),
      channels(channels),
      width(width),
      height(height),
      values(steps * channels * width * height) {
    std::mt19937 generator(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    for (auto& value : values) {
        value = normal(generator);
    }
}

std::vector<std::vector<float>> NoiseField::window(int left, int top, size_t windowWidth, size_t windowHeight) const {
    auto wrap = [](int value, size_t size) {
        const int modulo = value % int(size);
        return size_t(modulo < 0 ? modulo + int(size) : modulo);
    };
    std::vector<std::vector<float>> result(steps, std::vector<float>(channels * windowWidth * windowHeight));
    for (size_t step = 0; step < steps; ++step) {
        for (size_t channel = 0; channel < channels; ++channel) {
            const float* source = values.data() + (step * channels + channel) * width * height;
            for (size_t y = 0; y < windowHeight; ++y) {
                const size_t row = wrap(top + int(y), height);
                for (size_t x = 0; x < windowWidth; ++x) {
                    result[step][(channel * windowHeight + y) * windowWidth + x] =
                        source[row * width + wrap(left + int(x), width)];
                }
            }
        }
    }
    return result;
}

} // namespace klartraum
