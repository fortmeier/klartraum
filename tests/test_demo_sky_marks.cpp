// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - markStraightAheadIsCentred: a light ahead of the camera stands in the middle of the frame's width, between the
 *   horizon and the zenith rows
 * - markFollowsAzimuthAndElevation: a light more to the right stands more to the right, a higher one higher, even for a
 *   camera looking down
 * - markBehindOrBelowIsHidden: a light behind the camera or below the horizon is not shown
 * - sunIsADiscWithRays: on white, the sun's middle and a ray are black, the gap between two rays stays white
 * - marksStayBehindTheScene: pixels that are not the background colour are left as they are
 * - moonIsACrescent: on black, the moon's lit side is white, its middle on the other side stays black
 **/

#include <gtest/gtest.h>

#include <cmath>

#include <glm/gtc/matrix_transform.hpp>

#include "klartraum/demo_sky_marks.hpp"

using namespace klartraum;

namespace {

constexpr uint32_t kWidth = 320, kHeight = 180;

// A camera at the origin looking along -z, tilted down by `down` radians.
glm::mat4 viewLookingDown(float down) {
    return glm::lookAt(glm::vec3(0.0f), glm::vec3(0.0f, -std::sin(down), -std::cos(down)), glm::vec3(0.0f, 1.0f, 0.0f));
}

// A direction at `azimuth` radians to the right of -z and `elevation` above the horizon.
glm::vec3 towards(float azimuth, float elevation) {
    return glm::vec3(std::sin(azimuth) * std::cos(elevation), std::sin(elevation),
                     -std::cos(azimuth) * std::cos(elevation));
}

std::vector<uint8_t> greyImage(uint8_t grey) {
    std::vector<uint8_t> bgra(4 * kWidth * kHeight, grey);
    for (size_t i = 3; i < bgra.size(); i += 4)
        bgra[i] = 255;
    return bgra;
}

int brightness(const std::vector<uint8_t>& bgra, int x, int y) {
    const size_t i = 4 * (size_t(y) * kWidth + size_t(x));
    return (int(bgra[i]) + bgra[i + 1] + bgra[i + 2]) / 3;
}

} // namespace

TEST(DemoSkyMarks, markStraightAheadIsCentred) {
    const SkyMark mark = skyMarkAt(viewLookingDown(0.0f), towards(0.0f, 0.5f), kWidth, kHeight);
    ASSERT_TRUE(mark.visible);
    EXPECT_NEAR(mark.pixel.x, 0.5f * kWidth, 1.0f);
    EXPECT_GT(mark.pixel.y, 0.08f * kHeight);
    EXPECT_LT(mark.pixel.y, 0.5f * kHeight);
}

TEST(DemoSkyMarks, markFollowsAzimuthAndElevation) {
    const glm::mat4 view = viewLookingDown(0.4f);
    const SkyMark ahead = skyMarkAt(view, towards(0.0f, 0.3f), kWidth, kHeight);
    const SkyMark right = skyMarkAt(view, towards(0.6f, 0.3f), kWidth, kHeight);
    const SkyMark higher = skyMarkAt(view, towards(0.0f, 0.9f), kWidth, kHeight);
    ASSERT_TRUE(ahead.visible && right.visible && higher.visible);
    EXPECT_GT(right.pixel.x, ahead.pixel.x + 0.15f * kWidth);
    EXPECT_NEAR(right.pixel.y, ahead.pixel.y, 1.0f);
    EXPECT_LT(higher.pixel.y, ahead.pixel.y - 0.1f * kHeight);
}

TEST(DemoSkyMarks, markBehindOrBelowIsHidden) {
    const glm::mat4 view = viewLookingDown(0.3f);
    EXPECT_FALSE(skyMarkAt(view, towards(3.0f, 0.5f), kWidth, kHeight).visible);
    EXPECT_FALSE(skyMarkAt(view, towards(0.0f, -0.2f), kWidth, kHeight).visible);
}

TEST(DemoSkyMarks, sunIsADiscWithRays) {
    std::vector<uint8_t> image = greyImage(255);
    const glm::vec2 centre(160.0f, 90.0f);
    const float radius = 12.0f;
    drawSunMark(image, kWidth, kHeight, centre, radius, glm::vec3(0.0f), 1.0f, 1.0f);
    EXPECT_LT(brightness(image, 160, 90), 10);
    // A ray straight to the right, between 1.35 and 2 radii out; the gap between it and the next one.
    EXPECT_LT(brightness(image, 160 + int(1.65f * radius), 90), 60);
    const glm::vec2 gap = centre + 1.65f * radius * glm::vec2(std::cos(0.39f), std::sin(0.39f));
    EXPECT_GT(brightness(image, int(gap.x), int(gap.y)), 240);
    // Outside the sun.
    EXPECT_EQ(brightness(image, 160 + int(3.0f * radius), 90), 255);
}

TEST(DemoSkyMarks, marksStayBehindTheScene) {
    std::vector<uint8_t> image = greyImage(255);
    // A grey block of "scene" over the sun's middle.
    for (int y = 85; y < 95; ++y)
        for (int x = 155; x < 165; ++x)
            for (int c = 0; c < 3; ++c)
                image[4 * (size_t(y) * kWidth + size_t(x)) + size_t(c)] = 128;
    drawSunMark(image, kWidth, kHeight, glm::vec2(160.0f, 90.0f), 12.0f, glm::vec3(0.0f), 1.0f, 1.0f);
    EXPECT_EQ(brightness(image, 160, 90), 128);
    EXPECT_LT(brightness(image, 168, 90), 10);
}

TEST(DemoSkyMarks, moonIsACrescent) {
    std::vector<uint8_t> image = greyImage(0);
    const float radius = 12.0f;
    // Lit towards the lower right.
    const glm::vec2 lit = glm::normalize(glm::vec2(1.0f, 1.0f));
    drawMoonMark(image, kWidth, kHeight, glm::vec2(160.0f, 90.0f), radius, lit, glm::vec3(1.0f), 1.0f, 0.0f);
    const glm::vec2 litSide = glm::vec2(160.0f, 90.0f) + 0.8f * radius * lit;
    const glm::vec2 darkSide = glm::vec2(160.0f, 90.0f) - 0.3f * radius * lit;
    EXPECT_GT(brightness(image, int(litSide.x), int(litSide.y)), 200);
    EXPECT_LT(brightness(image, int(darkSide.x), int(darkSide.y)), 20);
    EXPECT_EQ(brightness(image, 160 + int(2.0f * radius), 90), 0);
}
