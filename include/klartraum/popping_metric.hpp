// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_POPPING_METRIC_HPP
#define KLARTRAUM_POPPING_METRIC_HPP

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

struct PoppingSummary {
    float max = 0.0f;
    float median = 0.0f;
    float mean = 0.0f;
    size_t worst = 0;   ///< index of the largest residual
    float spike = 0.0f; ///< max / median (0 when the median is 0)
};

/// The largest, median and mean residual of a run (empty: all 0).
PoppingSummary summarizePopping(const std::vector<float>& residuals);

} // namespace klartraum

#endif // KLARTRAUM_POPPING_METRIC_HPP
