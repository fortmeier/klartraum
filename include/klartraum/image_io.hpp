// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_IMAGE_IO_HPP
#define KLARTRAUM_IMAGE_IO_HPP

#include <cstdint>
#include <filesystem>
#include <vector>

namespace klartraum {

/**
 * @brief Writes an 8-bit image of @p channels (1 grey, 2 grey + alpha, 3 RGB or 4 RGBA) interleaved channels as
 * PNG (deflated by zlib, adaptive row filters) or binary PPM/PGM, chosen by the extension of @p path (.png, .ppm for
 * RGB, .pgm for grey).
 * @throws std::invalid_argument For other extensions or channel counts the format cannot hold.
 * @throws std::runtime_error If the file cannot be written.
 */
void writeImage(const std::filesystem::path& path, const uint8_t* pixels, uint32_t width, uint32_t height,
                uint32_t channels);

/** @brief writeImage() of a BGRA readback (e.g. of an OffscreenTarget), written as RGB. */
void writeImageBgra(const std::filesystem::path& path, const std::vector<uint8_t>& bgra, uint32_t width,
                    uint32_t height);

} // namespace klartraum

#endif // KLARTRAUM_IMAGE_IO_HPP
