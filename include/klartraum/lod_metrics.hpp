// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_LOD_METRICS_HPP
#define KLARTRAUM_LOD_METRICS_HPP

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace klartraum {

/// The luminance in [0, 1] of every pixel of a BGRA8 image (Rec. 709 weights).
std::vector<float> imageLuminance(const std::vector<uint8_t>& bgra);

struct PoppingConfig {
    uint32_t block = 16;    ///< side of the square blocks in pixels; a partial block at the edges is left out
    float minLight = 0.01f; ///< blocks darker than this (mean luminance) in all three frames are left out
};

/**
 * @brief How much a frame jumps out of a smooth motion: the mean over the lit blocks of
 *        |current - (previous + next) / 2| of the blocks' mean luminances.
 *
 * With a camera moving smoothly a block's light changes smoothly, so a frame lies near the mean of its
 * neighbours; a sudden change between two frames (a level of detail switching, a tree appearing) puts half its
 * size into the residual of both. Taking the blocks' means first keeps fine texture moving within a block (leaves
 * and twigs sliding by a few pixels per frame) out of the residual, while changes of coverage and brightness, what
 * a pop changes, stay in it. The mean is taken over the blocks lit in any of the three frames, so a pop of a whole
 * object weighs the same at any size on screen. Returns 0 without lit blocks. The frames are luminance images
 * (imageLuminance()) of width x height pixels.
 */
float poppingResidual(const std::vector<float>& previous, const std::vector<float>& current,
                      const std::vector<float>& next, uint32_t width, uint32_t height, PoppingConfig config = {});

/// poppingResidual() over a stream of frames: push() returns the residual of the frame before the pushed one, once
/// it has neighbours on both sides.
class PoppingMeter {
public:
    PoppingMeter(uint32_t width, uint32_t height, PoppingConfig config = {});

    std::optional<float> push(std::vector<float> frame);
    /// Forgets the frames so far, e.g. at a cut.
    void reset();

private:
    uint32_t width;
    uint32_t height;
    PoppingConfig config;
    std::vector<std::vector<float>> frames; // the last two
};

struct SeriesSummary {
    float max = 0.0f;
    float median = 0.0f;
    float mean = 0.0f;
    size_t worst = 0;   ///< index of the largest value
    float spike = 0.0f; ///< max / median (0 when the median is 0)
};

/// The largest, median and mean of a run of per-frame values, e.g. residuals or errors (empty: all 0).
SeriesSummary summarizeSeries(const std::vector<float>& values);

struct FidelityConfig {
    uint32_t block = 4;     ///< side of the square blocks in pixels; a partial block at the edges is left out
    float minLight = 0.01f; ///< blocks darker than this (mean of the RGB means) in both images are left out
};

/**
 * @brief How far an image lies from a reference of the same view (e.g. the same tree without levels of detail): the
 *        mean over the lit blocks of the absolute difference of the blocks' RGB means, averaged over the channels,
 *        in [0, 1].
 *
 * A block counts when it is lit in either image, so a part missing from the image weighs as much as a part too
 * many; blocks dark in both are left out, which makes the error independent of how much of the screen the object
 * covers. Detail finer than a block that keeps its mean light does not count. Returns 0 without lit blocks. The
 * images are BGRA8 of width x height pixels.
 */
float fidelityError(const std::vector<uint8_t>& image, const std::vector<uint8_t>& reference, uint32_t width,
                    uint32_t height, FidelityConfig config = {});

struct DetailConfig {
    uint32_t block = 8;     ///< side of the square blocks in pixels; a partial block at the edges is left out
    float minLight = 0.01f; ///< blocks darker than this (mean luminance) in both images are left out
    float floor = 1e-4f;    ///< added to both energies (luminance squared), so that flat blocks compare as equal
};

/**
 * @brief How far an image's detail lies from a reference's: per block, the energy of the luminance's Laplacian
 *        (its mean square, the fine structure), compared as |log2(image's / reference's)|, averaged over the lit
 *        blocks.
 *
 * 0 means as much fine structure as the reference, 1 twice or half as much (in energy), whichever way: a blurred
 * image scores like a noisy one. Constant offsets do not count, so colour shifts are left to fidelityError(); the
 * two together tell colour from sharpness. The images are BGRA8 of width x height pixels.
 */
float detailError(const std::vector<uint8_t>& image, const std::vector<uint8_t>& reference, uint32_t width,
                  uint32_t height, DetailConfig config = {});

} // namespace klartraum

#endif // KLARTRAUM_LOD_METRICS_HPP
