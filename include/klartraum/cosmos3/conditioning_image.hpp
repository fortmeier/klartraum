// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_COSMOS3_CONDITIONING_IMAGE_HPP
#define KLARTRAUM_COSMOS3_CONDITIONING_IMAGE_HPP

#include <cstdint>
#include <filesystem>
#include <vector>

namespace klartraum {

/** @brief An 8-bit RGB image, rows top to bottom, pixels interleaved. */
struct RgbImage {
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<uint8_t> pixels; ///< width * height * 3 bytes
};

/**
 * @brief Reads a binary (P6) PPM with a maximum value of 255.
 * @throws std::runtime_error If the file cannot be read or is not such a PPM.
 */
RgbImage readPpm(const std::filesystem::path& path);

/**
 * @brief Prepares an image as the Cosmos3 conditioning frame.
 *
 * Reproduces diffusers' `_preprocess_conditioning_image` for 8-bit input: the
 * image is scaled to cover @p width x @p height (sizes rounded up), resampled
 * with torch's antialiased bilinear filter, centre-cropped, rounded to integer
 * levels, and mapped to [-1, 1].
 *
 * @return Planar float32 tensor [3, height, width].
 */
std::vector<float> preprocessConditioningImage(const RgbImage& image, uint32_t width, uint32_t height);

} // namespace klartraum

#endif // KLARTRAUM_COSMOS3_CONDITIONING_IMAGE_HPP
