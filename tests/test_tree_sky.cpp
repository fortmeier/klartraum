// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - noonIsHigherInSummer: at noon the sun stands 90 - latitude degrees high at the equinoxes, higher in summer and
 *   lower in winter, in the south (+z)
 * - daysAreLongerInSummer: the sun is up about half the day at the equinoxes, longer in summer, shorter in winter
 * - sunRisesInTheEast: in the morning the sun is in the east (+x), in the evening in the west, below the horizon
 *   at midnight
 * - nightIsMoonlit: at midnight the moon casts the light, dim and bluish, the exposure is higher and the lantern
 *   glows; at noon the sun casts it and the lantern does not glow
 * - lightChangesSmoothly: the light (direction times intensity and colour), the ambient light and the exposure
 *   change only a little between frames of a 10 s day, also when the sun hands over to the moon
 * - applyWritesTheSettings: applyTreeSky sets light, colour, ambient, exposure, point light and tint
 * - daysShowOneSeasonEach: four days of 10 s each, every one showing its season while the sun is up; the season
 *   changes only while the sun is down, and the last night leads back to the first season
 * - seasonsChangeSmoothlyForAnyDawn: with days starting anywhere from midnight to the default dawn, the season changes
 *   forwards and smoothly across every midnight and shows the day's season while the sun is up
 **/

#include <gtest/gtest.h>

#include <cmath>

#include "klartraum/tree/tree_sky.hpp"

using namespace klartraum;

TEST(TreeSky, noonIsHigherInSummer) {
    const float spring = glm::degrees(treeSkyAt(0.5f, 1.0f).sunElevation);
    const float summer = glm::degrees(treeSkyAt(0.5f, 2.0f).sunElevation);
    const float winter = glm::degrees(treeSkyAt(0.5f, 0.0f).sunElevation);
    EXPECT_NEAR(spring, 42.0f, 0.5f);
    EXPECT_NEAR(summer, 42.0f + 23.44f, 0.5f);
    EXPECT_NEAR(winter, 42.0f - 23.44f, 0.5f);
    EXPECT_GT(treeSkyAt(0.5f, 2.0f).sunDirection.z, 0.0f);
}

TEST(TreeSky, daysAreLongerInSummer) {
    auto dayLength = [](float season) {
        int up = 0;
        for (int i = 0; i < 1000; ++i) {
            up += treeSkyAt(float(i) / 1000.0f, season).sunElevation > 0.0f ? 1 : 0;
        }
        return float(up) / 1000.0f;
    };
    EXPECT_NEAR(dayLength(1.0f), 0.5f, 0.01f);
    EXPECT_NEAR(dayLength(3.0f), 0.5f, 0.01f);
    EXPECT_GT(dayLength(2.0f), 0.6f);
    EXPECT_LT(dayLength(0.0f), 0.4f);
}

TEST(TreeSky, sunRisesInTheEast) {
    EXPECT_GT(treeSkyAt(0.3f, 2.0f).sunDirection.x, 0.5f);
    EXPECT_LT(treeSkyAt(0.7f, 2.0f).sunDirection.x, -0.5f);
    EXPECT_LT(treeSkyAt(0.0f, 2.0f).sunElevation, 0.0f);
}

TEST(TreeSky, nightIsMoonlit) {
    const TreeSky night = treeSkyAt(0.0f, 1.0f);
    EXPECT_EQ(night.lightDirection, night.moonDirection);
    EXPECT_GT(night.lightIntensity, 0.05f);
    EXPECT_LT(night.lightIntensity, 0.5f);
    EXPECT_GT(night.lightColor.b, night.lightColor.r);
    EXPECT_EQ(night.lanternGlow, 1.0f);
    const TreeSky noon = treeSkyAt(0.5f, 1.0f);
    EXPECT_EQ(noon.lightDirection, noon.sunDirection);
    EXPECT_GT(noon.lightIntensity, 1.0f);
    EXPECT_EQ(noon.lanternGlow, 0.0f);
    EXPECT_GT(night.exposure, 2.0f * noon.exposure);
}

TEST(TreeSky, lightChangesSmoothly) {
    for (float season : {0.0f, 1.0f, 2.0f}) {
        TreeSky previous = treeSkyAt(0.0f, season);
        // One frame of a 10 s day at 30 fps.
        const float step = 1.0f / 300.0f;
        for (float t = step; t <= 1.0f; t += step) {
            const TreeSky sky = treeSkyAt(t, season);
            const glm::vec3 light = sky.lightDirection * sky.lightIntensity * sky.lightColor;
            const glm::vec3 before = previous.lightDirection * previous.lightIntensity * previous.lightColor;
            // In a 10 s day, dawn and dusk (sun from 6 below to 14 degrees above the horizon) take about half a
            // second, 16 frames: the light changes by up to a few percent of its range per frame then, no more.
            EXPECT_LT(glm::length(light - before), 0.15f) << "season " << season << " t " << t;
            EXPECT_LT(glm::length(sky.sky - previous.sky), 0.05f) << t;
            EXPECT_LT(std::abs(sky.exposure - previous.exposure), 0.2f) << t;
            previous = sky;
        }
    }
}

TEST(TreeSky, applyWritesTheSettings) {
    TreeRenderSettings settings = defaultTreeRenderSettings({64, 64});
    const TreeSky sky = treeSkyAt(0.02f, 2.0f);
    applyTreeSky(settings, sky, glm::vec3(1.0f, 2.0f, 3.0f), 0.5f);
    EXPECT_EQ(glm::vec3(settings.sun), sky.lightDirection);
    EXPECT_EQ(settings.sun.w, sky.lightIntensity);
    EXPECT_EQ(glm::vec3(settings.lightColor), sky.lightColor);
    EXPECT_EQ(glm::vec3(settings.sky), sky.sky);
    EXPECT_EQ(settings.sky.w, sky.exposure);
    EXPECT_EQ(glm::vec3(settings.pointLight), glm::vec3(1.0f, 2.0f, 3.0f));
    EXPECT_GT(settings.pointColor.r, settings.pointColor.b);
    EXPECT_EQ(glm::vec3(settings.capturedTint), sky.capturedTint);
}

TEST(TreeSky, daysShowOneSeasonEach) {
    const BonsaiDays days;
    EXPECT_EQ(days.duration(), 40.0f);
    const float frame = 1.0f / 30.0f;
    float previous = days.seasonAt(0.0f);
    for (float t = 0.0f; t < days.duration(); t += frame) {
        const uint32_t day = days.dayAt(t);
        const float season = days.seasonAt(t);
        const TreeSky sky = treeSkyAt(days.timeOfDayAt(t), season);
        if (sky.sunElevation > 0.0f) {
            EXPECT_NEAR(season, tree::treeSeasonWrap(days.seasons[day]), 1e-5f) << "t " << t;
        }
        // Forwards and smoothly (across the wrap from winter to spring).
        const float change = tree::treeSeasonWrap(season - previous + 2.0f) - 2.0f;
        EXPECT_GE(change, -1e-5f) << t;
        EXPECT_LT(change, 0.05f) << t;
        previous = season;
    }
    EXPECT_NEAR(days.seasonAt(days.duration() - frame), days.seasonAt(0.0f), 0.01f);
    EXPECT_NEAR(days.timeOfDayAt(0.0f), days.dawn, 1e-6f);
}

TEST(TreeSky, seasonsChangeSmoothlyForAnyDawn) {
    for (const float dawn : {0.0f, 0.05f, 0.12f}) {
        SCOPED_TRACE("dawn " + std::to_string(dawn));
        BonsaiDays days;
        days.secondsPerDay = 3.5f;
        days.dawn = dawn;
        const float frame = 1.0f / 30.0f;
        float previous = days.seasonAt(0.0f);
        float largest = 0.0f;
        for (float t = frame; t < days.duration(); t += frame) {
            const float season = days.seasonAt(t);
            const TreeSky sky = treeSkyAt(days.timeOfDayAt(t), season);
            if (sky.sunElevation > 0.0f)
                EXPECT_NEAR(season, tree::treeSeasonWrap(days.seasons[days.dayAt(t)]), 1e-5f) << "t " << t;
            const float change = tree::treeSeasonWrap(season - previous + 2.0f) - 2.0f;
            EXPECT_GE(change, -1e-5f) << "t " << t;
            largest = std::max(largest, change);
            previous = season;
        }
        // About 0.2 days for a change of up to 1.25 seasons, at 105 frames per day.
        EXPECT_LT(largest, 0.15f);
    }
}
