// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - luminanceOfColours: imageLuminance() of white, black and pure green BGRA pixels is 1, 0 and 0.7152
 * - stillFramesDoNotPop: three identical lit frames have a residual of 0
 * - linearFadeDoesNotPop: a frame halfway between its neighbours (a linear fade) has a residual of 0
 * - textureMovingWithinBlocksDoesNotPop: stripes of 1 px moving by a pixel per frame change every pixel but no
 *   block's mean, so the residual is 0
 * - stepPopsOnBothSides: a lit block jumping by 0.4 between two frames gives 0.2 to the frames on both sides of the
 *   jump and 0 to the others
 * - darkBlocksAreLeftOut: blocks dark in all three frames do not count, so a pop of one lit block weighs the same
 *   whatever the size of the image; all dark gives 0
 * - partialBlocksAreLeftOut: pixels outside whole blocks do not change the residual
 * - framesMustMatchTheSize: frames of the wrong size are rejected
 * - meterStreamsTheResiduals: PoppingMeter returns nothing for the first two frames, then each middle frame's
 *   residual, and starts over after reset()
 * - summaryOfARun: summarizeSeries() gives the largest, median and mean value, the worst frame and max / median;
 *   an empty run gives zeros
 * - identicalImagesAreFaithful: an image against itself has a fidelity error of 0
 * - fidelityIsTheMeanBlockDifference: one 4 x 4 block brighter by 0.4 in one channel out of two lit blocks gives
 *   0.4 / 3 / 2; a difference within a block that keeps its mean gives 0
 * - fidelityCountsBlocksLitInEitherImage: a block lit only in the reference (a part missing) or only in the image (a
 *   part too many) counts, blocks dark in both do not; both all dark gives 0
 * - fidelityImagesMustMatchTheSize: images of the wrong size are rejected
 **/

#include <gtest/gtest.h>

#include <cmath>

#include "klartraum/lod_metrics.hpp"

using namespace klartraum;

namespace {

// A width x height luminance image of `value`, with the 16 x 16 block at (bx, by) set to `block` when given.
std::vector<float> image(uint32_t width, uint32_t height, float value, int bx = -1, int by = -1, float block = 0.0f) {
    std::vector<float> l(size_t(width) * height, value);
    if (bx >= 0) {
        for (uint32_t y = uint32_t(by) * 16; y < uint32_t(by + 1) * 16; ++y) {
            for (uint32_t x = uint32_t(bx) * 16; x < uint32_t(bx + 1) * 16; ++x) {
                l[size_t(y) * width + x] = block;
            }
        }
    }
    return l;
}

} // namespace

TEST(LodMetricsTest, luminanceOfColours) {
    const std::vector<uint8_t> bgra = {255, 255, 255, 255, 0, 0, 0, 255, 0, 255, 0, 255};
    const std::vector<float> l = imageLuminance(bgra);
    ASSERT_EQ(l.size(), 3u);
    EXPECT_NEAR(l[0], 1.0f, 1e-5f);
    EXPECT_NEAR(l[1], 0.0f, 1e-6f);
    EXPECT_NEAR(l[2], 0.7152f, 1e-5f);
}

TEST(LodMetricsTest, stillFramesDoNotPop) {
    const std::vector<float> a = image(64, 32, 0.5f);
    EXPECT_FLOAT_EQ(poppingResidual(a, a, a, 64, 32), 0.0f);
}

TEST(LodMetricsTest, linearFadeDoesNotPop) {
    EXPECT_NEAR(poppingResidual(image(64, 32, 0.2f), image(64, 32, 0.4f), image(64, 32, 0.6f), 64, 32), 0.0f, 1e-6f);
}

TEST(LodMetricsTest, stepPopsOnBothSides) {
    // A lit image whose block (1, 0) jumps from 0.3 to 0.7 between frames 2 and 3; all blocks are lit.
    std::vector<std::vector<float>> frames;
    for (int t = 0; t < 6; ++t) {
        frames.push_back(image(64, 32, 0.5f, 1, 0, t <= 2 ? 0.3f : 0.7f));
    }
    const float blocks = 8.0f; // 4 x 2 blocks of 16 px
    for (int t = 1; t + 1 < 6; ++t) {
        const float r = poppingResidual(frames[t - 1], frames[t], frames[t + 1], 64, 32);
        const float expected = (t == 2 || t == 3) ? 0.2f / blocks : 0.0f;
        EXPECT_NEAR(r, expected, 1e-6f) << t;
    }
}

TEST(LodMetricsTest, textureMovingWithinBlocksDoesNotPop) {
    // Vertical stripes of 1 px moving by a pixel per frame: every pixel changes, no block's mean does.
    auto stripes = [](int shift) {
        std::vector<float> l(64 * 32);
        for (uint32_t y = 0; y < 32; ++y) {
            for (uint32_t x = 0; x < 64; ++x) {
                l[size_t(y) * 64 + x] = (x + uint32_t(shift)) % 2 == 0 ? 0.2f : 0.8f;
            }
        }
        return l;
    };
    EXPECT_NEAR(poppingResidual(stripes(0), stripes(1), stripes(2), 64, 32), 0.0f, 1e-6f);
}

TEST(LodMetricsTest, darkBlocksAreLeftOut) {
    // One lit block jumps by 0.4: its residual is 0.2 in a small image and in a large one.
    for (uint32_t size : {32u, 128u}) {
        const std::vector<float> before = image(size, size, 0.0f, 0, 0, 0.3f);
        const std::vector<float> after = image(size, size, 0.0f, 0, 0, 0.7f);
        EXPECT_NEAR(poppingResidual(before, before, after, size, size), 0.2f, 1e-6f) << size;
    }
    const std::vector<float> dark = image(64, 64, 0.005f);
    EXPECT_FLOAT_EQ(poppingResidual(dark, dark, image(64, 64, 0.0f), 64, 64), 0.0f);
}

TEST(LodMetricsTest, partialBlocksAreLeftOut) {
    // 40 x 20 pixels: two whole blocks; the jump happens only in the leftover columns and rows.
    std::vector<float> a(40 * 20, 0.5f), b = a;
    for (uint32_t y = 0; y < 20; ++y) {
        for (uint32_t x = 0; x < 40; ++x) {
            if (x >= 32 || y >= 16) {
                b[size_t(y) * 40 + x] = 1.0f;
            }
        }
    }
    EXPECT_FLOAT_EQ(poppingResidual(a, a, b, 40, 20), 0.0f);
}

TEST(LodMetricsTest, framesMustMatchTheSize) {
    const std::vector<float> a = image(32, 32, 0.5f);
    EXPECT_THROW(poppingResidual(a, a, image(32, 16, 0.5f), 32, 32), std::invalid_argument);
}

TEST(LodMetricsTest, meterStreamsTheResiduals) {
    PoppingMeter meter(32, 32);
    EXPECT_FALSE(meter.push(image(32, 32, 0.5f)).has_value());
    EXPECT_FALSE(meter.push(image(32, 32, 0.5f)).has_value());
    // The second frame between 0.5 and 0.9: residual |0.5 - 0.7| = 0.2.
    const std::optional<float> r = meter.push(image(32, 32, 0.9f));
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(*r, 0.2f, 1e-6f);
    // The third frame (0.9) between 0.5 and 0.9: 0.2 again.
    const std::optional<float> r2 = meter.push(image(32, 32, 0.9f));
    ASSERT_TRUE(r2.has_value());
    EXPECT_NEAR(*r2, 0.2f, 1e-6f);
    meter.reset();
    EXPECT_FALSE(meter.push(image(32, 32, 0.5f)).has_value());
    EXPECT_FALSE(meter.push(image(32, 32, 0.5f)).has_value());
}

TEST(LodMetricsTest, summaryOfARun) {
    const SeriesSummary s = summarizeSeries({0.1f, 0.4f, 0.2f, 0.1f, 0.2f});
    EXPECT_FLOAT_EQ(s.max, 0.4f);
    EXPECT_EQ(s.worst, 1u);
    EXPECT_FLOAT_EQ(s.median, 0.2f);
    EXPECT_NEAR(s.mean, 0.2f, 1e-6f);
    EXPECT_FLOAT_EQ(s.spike, 2.0f);
    const SeriesSummary even = summarizeSeries({0.1f, 0.3f});
    EXPECT_FLOAT_EQ(even.median, 0.2f);
    const SeriesSummary empty = summarizeSeries({});
    EXPECT_FLOAT_EQ(empty.max, 0.0f);
    EXPECT_FLOAT_EQ(empty.spike, 0.0f);
}

namespace {

// A width x height BGRA image of grey `value` in [0, 1].
std::vector<uint8_t> grey(uint32_t width, uint32_t height, float value) {
    const uint8_t v = uint8_t(std::lround(value * 255.0f));
    std::vector<uint8_t> bgra(size_t(width) * height * 4, v);
    for (size_t i = 3; i < bgra.size(); i += 4) {
        bgra[i] = 255;
    }
    return bgra;
}

// Sets channel c (0 b, 1 g, 2 r) of the pixels [x0, x1) x [y0, y1) to `value` in [0, 1].
void paint(std::vector<uint8_t>& bgra, uint32_t width, uint32_t x0, uint32_t x1, uint32_t y0, uint32_t y1, int c,
           float value) {
    for (uint32_t y = y0; y < y1; ++y) {
        for (uint32_t x = x0; x < x1; ++x) {
            bgra[(size_t(y) * width + x) * 4 + c] = uint8_t(std::lround(value * 255.0f));
        }
    }
}

} // namespace

TEST(LodMetricsTest, identicalImagesAreFaithful) {
    std::vector<uint8_t> a = grey(16, 8, 0.4f);
    paint(a, 16, 0, 4, 0, 4, 2, 1.0f);
    EXPECT_FLOAT_EQ(fidelityError(a, a, 16, 8), 0.0f);
}

TEST(LodMetricsTest, fidelityIsTheMeanBlockDifference) {
    // Two lit 4 x 4 blocks (8 x 4 pixels); the image's first block is brighter by 0.4 in red.
    const std::vector<uint8_t> reference = grey(8, 4, 0.4f);
    std::vector<uint8_t> image = reference;
    paint(image, 8, 0, 4, 0, 4, 2, 0.8f);
    EXPECT_NEAR(fidelityError(image, reference, 8, 4), 0.4f / 3.0f / 2.0f, 1e-3f);
    // Half of a block brighter by 0.4 and the other half darker by 0.4: the block's mean stays.
    std::vector<uint8_t> shuffled = reference;
    paint(shuffled, 8, 0, 2, 0, 4, 1, 0.8f);
    paint(shuffled, 8, 2, 4, 0, 4, 1, 0.0f);
    EXPECT_NEAR(fidelityError(shuffled, reference, 8, 4), 0.0f, 1e-6f);
}

TEST(LodMetricsTest, fidelityCountsBlocksLitInEitherImage) {
    // Four blocks over black: the reference has blocks 0 and 1 lit, the image blocks 1 and 2; block 3 is dark in
    // both. Blocks 0 and 2 differ by 0.6 in every channel, block 1 matches: (0.6 + 0 + 0.6) / 3.
    std::vector<uint8_t> reference = grey(16, 4, 0.0f), image = grey(16, 4, 0.0f);
    for (int c = 0; c < 3; ++c) {
        paint(reference, 16, 0, 8, 0, 4, c, 0.6f);
        paint(image, 16, 4, 12, 0, 4, c, 0.6f);
    }
    EXPECT_NEAR(fidelityError(image, reference, 16, 4), 1.2f / 3.0f, 1e-3f);
    EXPECT_FLOAT_EQ(fidelityError(grey(16, 4, 0.0f), grey(16, 4, 0.0f), 16, 4), 0.0f);
}

TEST(LodMetricsTest, fidelityImagesMustMatchTheSize) {
    EXPECT_THROW(fidelityError(grey(8, 4, 0.5f), grey(4, 4, 0.5f), 8, 4), std::invalid_argument);
}
