// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - chaptersFollowEachOther: chapter starts, lengths and the frame count follow the chapter lengths at
 *   the frame rate, and every frame belongs to the chapter whose time span contains it
 * - morphHoldsThenRuns: the morph progress is 0 during the title hold, rises during the morph and stays
 *   1 afterwards
 * - orbitStartsWithMorphAndKeepsItsSpeed: the orbit angle is 0 during the title hold, its step grows
 *   evenly during the morph and then stays at one turn per orbit period, across all chapter changes
 * - orbitSlowsDownSmoothly: from the slow chapter on, the orbit's step per frame eases from one turn per orbit
 *   period down to one turn per slow period within the transition time, without a jump, and the angle is the sum
 *   of the steps
 * - pushInEasesBetweenDistances: the camera distance stays at the start distance before the push-in,
 *   moves monotonically to the end distance during it and stays there
 * - captionFadesInAndOut: a caption is invisible outside its window, fully visible in its hold, and
 *   fades monotonically in between
 * - leftAlignedLinesStartTogether: with left alignment all lines of a block start at the same x,
 *   centred alignment centres each line
 * - blendCaptionSetsCoveredPixels: white text at full opacity on black turns exactly the covered pixels
 *   white (the frame equals the coverage), zero opacity leaves the frame unchanged, and half opacity
 *   gives half the brightness
 **/

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "klartraum/demo_timeline.hpp"
#include "klartraum/rendering/text_gaussians.hpp"

using namespace klartraum;

namespace {

std::shared_ptr<Font> loadFont(const std::string& name) {
    try {
        return std::make_shared<Font>("fonts/" + name + ".ttf");
    } catch (const std::exception&) {
        return nullptr;
    }
}

#define REQUIRE_FONT(var, name)                                                                                        \
    auto var = loadFont(name);                                                                                         \
    if (!var) {                                                                                                        \
        GTEST_SKIP() << "font not available: " << name << " (configure with KLARTRAUM_DOWNLOAD_FONTS=ON)";             \
    }

DemoTimeline exampleTimeline() {
    DemoTimeline timeline;
    timeline.fps = 30.0f;
    timeline.chapterSeconds = {8.5f, 7.5f, 10.0f};
    timeline.titleSeconds = 2.0f;
    timeline.morphSeconds = 1.5f;
    timeline.secondsPerTurn = 15.0f;
    return timeline;
}

} // namespace

TEST(DemoTimeline, chaptersFollowEachOther) {
    const DemoTimeline timeline = exampleTimeline();
    EXPECT_EQ(timeline.frameCount(), 780u);
    EXPECT_EQ(timeline.chapterStart(0), 0u);
    EXPECT_EQ(timeline.chapterStart(1), 255u);
    EXPECT_EQ(timeline.chapterStart(2), 480u);
    EXPECT_EQ(timeline.chapterAt(0), 0);
    EXPECT_EQ(timeline.chapterAt(254), 0);
    EXPECT_EQ(timeline.chapterAt(255), 1);
    EXPECT_EQ(timeline.chapterAt(479), 1);
    EXPECT_EQ(timeline.chapterAt(480), 2);
    EXPECT_EQ(timeline.chapterAt(779), 2);
    EXPECT_FLOAT_EQ(timeline.timeInChapter(255), 0.0f);
    EXPECT_FLOAT_EQ(timeline.timeInChapter(270), 0.5f);
}

TEST(DemoTimeline, morphHoldsThenRuns) {
    const DemoTimeline timeline = exampleTimeline();
    for (uint32_t f = 0; f < 60; ++f) {
        EXPECT_EQ(timeline.morphProgress(f), 0.0f) << f;
    }
    float previous = 0.0f;
    for (uint32_t f = 61; f < 105; ++f) {
        const float p = timeline.morphProgress(f);
        EXPECT_GT(p, previous) << f;
        EXPECT_LT(p, 1.0f) << f;
        previous = p;
    }
    for (uint32_t f = 105; f < timeline.frameCount(); ++f) {
        EXPECT_EQ(timeline.morphProgress(f), 1.0f) << f;
    }
}

TEST(DemoTimeline, orbitStartsWithMorphAndKeepsItsSpeed) {
    const DemoTimeline timeline = exampleTimeline();
    for (uint32_t f = 0; f <= 60; ++f) {
        EXPECT_EQ(timeline.orbitAngle(f), 0.0) << f;
    }
    const double rate = 2.0 * 3.14159265358979323846 / 15.0 / 30.0; // per frame
    double previousStep = 0.0;
    for (uint32_t f = 61; f < timeline.frameCount(); ++f) {
        const double step = timeline.orbitAngle(f) - timeline.orbitAngle(f - 1);
        EXPECT_GT(step, 0.0) << f;
        EXPECT_GE(step, previousStep - 1e-12) << f;
        EXPECT_LE(step, rate + 1e-12) << f;
        if (f > 106) {
            EXPECT_NEAR(step, rate, 1e-9) << f;
        }
        previousStep = step;
    }
}

TEST(DemoTimeline, orbitSlowsDownSmoothly) {
    DemoTimeline timeline = exampleTimeline();
    timeline.slowChapter = 2;
    timeline.slowSecondsPerTurn = 60.0f;
    timeline.slowTransitionSeconds = 1.5f;
    const double fast = 2.0 * 3.14159265358979323846 / 15.0 / 30.0, slow = fast / 4.0;
    const uint32_t start = timeline.chapterStart(2);
    for (uint32_t f = 107; f <= start; ++f) {
        EXPECT_NEAR(timeline.orbitAngle(f) - timeline.orbitAngle(f - 1), fast, 1e-9) << f;
    }
    double previousStep = fast;
    for (uint32_t f = start + 1; f < timeline.frameCount(); ++f) {
        const double step = timeline.orbitAngle(f) - timeline.orbitAngle(f - 1);
        EXPECT_LE(step, previousStep + 1e-12) << f;
        EXPECT_GE(step, slow - 1e-12) << f;
        // The rate changes by at most a few percent of the fast rate per frame.
        EXPECT_LT(previousStep - step, 0.05 * fast) << f;
        if (f >= start + 46) {
            EXPECT_NEAR(step, slow, 1e-9) << f;
        }
        previousStep = step;
    }
}

TEST(DemoTimeline, pushInEasesBetweenDistances) {
    DemoTimeline timeline = exampleTimeline();
    timeline.startDistance = 0.65f;
    timeline.endDistance = 0.55f;
    timeline.pushInChapter = 1;
    const uint32_t first = timeline.chapterStart(1);
    const uint32_t last = timeline.chapterStart(2);
    for (uint32_t f = 0; f <= first; ++f) {
        EXPECT_FLOAT_EQ(timeline.cameraDistance(f), 0.65f) << f;
    }
    for (uint32_t f = first + 1; f < last; ++f) {
        EXPECT_LE(timeline.cameraDistance(f), timeline.cameraDistance(f - 1)) << f;
    }
    for (uint32_t f = last; f < timeline.frameCount(); ++f) {
        EXPECT_FLOAT_EQ(timeline.cameraDistance(f), 0.55f) << f;
    }
}

TEST(DemoTimeline, captionFadesInAndOut) {
    const DemoTimeline timeline = exampleTimeline();
    // Chapter 1 (7.5 s): visible from 0.5 s, fully from 1.0 s, until 0.5 s before its end.
    const CaptionWindow window{1, 0.5f, 0.5f, 0.5f};
    const uint32_t start = timeline.chapterStart(1);
    EXPECT_EQ(timeline.captionOpacity(window, start - 1), 0.0f);
    EXPECT_EQ(timeline.captionOpacity(window, start), 0.0f);
    EXPECT_EQ(timeline.captionOpacity(window, start + 15), 0.0f);
    EXPECT_GT(timeline.captionOpacity(window, start + 20), 0.0f);
    EXPECT_LT(timeline.captionOpacity(window, start + 20), 1.0f);
    EXPECT_EQ(timeline.captionOpacity(window, start + 30), 1.0f);
    EXPECT_EQ(timeline.captionOpacity(window, start + 195), 1.0f);
    EXPECT_EQ(timeline.captionOpacity(window, start + 210), 0.0f);
    EXPECT_EQ(timeline.captionOpacity(window, timeline.chapterStart(2)), 0.0f);
    float previous = 0.0f;
    for (uint32_t f = start; f <= start + 30; ++f) {
        EXPECT_GE(timeline.captionOpacity(window, f), previous) << f;
        previous = timeline.captionOpacity(window, f);
    }
    for (uint32_t f = start + 195; f <= start + 210; ++f) {
        EXPECT_LE(timeline.captionOpacity(window, f), previous) << f;
        previous = timeline.captionOpacity(window, f);
    }
}

TEST(DemoTimeline, leftAlignedLinesStartTogether) {
    REQUIRE_FONT(mono, "JetBrainsMono");
    TitleLayout layout;
    layout.lines.push_back({"a long first line", mono, 400.0f, 1.0f, 0.0f});
    layout.lines.push_back({"short", mono, 400.0f, 1.0f, 0.0f});
    layout.pixelsPerUnit = 32.0f;

    auto firstInkColumn = [](const CoverageMask& mask, int rowBegin, int rowEnd) {
        int column = mask.width;
        for (int y = rowBegin; y < rowEnd; ++y) {
            for (int x = 0; x < mask.width; ++x) {
                if (mask.at(x, y) >= 128) {
                    column = std::min(column, x);
                }
            }
        }
        return column;
    };
    layout.align = TextAlign::Left;
    const CoverageMask left = rasterizeTitle(layout);
    const int half = left.height / 2;
    // "a" and "s" both have a little side bearing; they start within two pixels.
    EXPECT_NEAR(firstInkColumn(left, 0, half), firstInkColumn(left, half, left.height), 2);

    // Centred, the shorter line starts half the 12 missing letters (0.6 em
    // each in this monospace font) further right: 3.6 em.
    layout.align = TextAlign::Centre;
    const CoverageMask centred = rasterizeTitle(layout);
    EXPECT_GT(firstInkColumn(centred, centred.height / 2, centred.height) -
                  firstInkColumn(centred, 0, centred.height / 2),
              int(3 * layout.pixelsPerUnit));
}

TEST(DemoTimeline, blendCaptionSetsCoveredPixels) {
    REQUIRE_FONT(mono, "JetBrainsMono");
    TitleLayout layout;
    layout.lines.push_back({"klartraum", mono, 400.0f, 24.0f, 0.0f});
    layout.pixelsPerUnit = 1.0f; // sizes in pixels
    const CoverageMask mask = rasterizeTitle(layout);
    const uint32_t width = uint32_t(mask.width) + 20, height = uint32_t(mask.height) + 10;

    std::vector<uint8_t> frame(size_t(width) * height * 4, 0);
    blendCaption(frame, width, height, mask, 10, 5, glm::vec3(1.0f), 1.0f);
    size_t wrong = 0;
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            const int mx = int(x) - 10, my = int(y) - 5;
            const uint8_t expected = (mx >= 0 && my >= 0 && mx < mask.width && my < mask.height) ? mask.at(mx, my) : 0;
            const size_t i = (size_t(y) * width + x) * 4;
            wrong += std::abs(int(frame[i]) - expected) > 1 || frame[i + 1] != frame[i] || frame[i + 2] != frame[i];
            wrong += frame[i + 3] != 0;
        }
    }
    EXPECT_EQ(wrong, 0u);

    std::vector<uint8_t> grey(size_t(width) * height * 4, 100);
    std::vector<uint8_t> unchanged = grey;
    blendCaption(unchanged, width, height, mask, 10, 5, glm::vec3(1.0f), 0.0f);
    EXPECT_EQ(unchanged, grey);

    std::vector<uint8_t> half(size_t(width) * height * 4, 0);
    blendCaption(half, width, height, mask, 10, 5, glm::vec3(1.0f), 0.5f);
    size_t far = 0;
    for (size_t i = 0; i < half.size(); i += 4) {
        far += std::abs(2 * int(half[i]) - int(frame[i])) > 2;
    }
    EXPECT_EQ(far, 0u);
}
