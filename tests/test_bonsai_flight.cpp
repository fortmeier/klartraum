// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - smootherstepEasesInAndOut: 0 below 0, 1 above 1, 1/2 at 1/2, with flat ends
 * - slerpDirectionStaysOnTheSphere: unit length all the way, a at 0, b at 1, and b when a and b agree
 * - flightStartsAtTheStart: at u = 0 the camera is at the start, looking at the focus, the near plane 0.05 units
 * - flightEndsAboveTheGround: at scale 1, every pose of the fly-in to the reference place stays above the soil; at
 *   u = 1 the camera is about `above` over the soil near the place and looks along the heading, panned by 35 deg
 * - forestApproachLiesInDenseForest: at scale 1 there are places to fly into, and the best one has forest (weight
 *   >= 0.9) within 30 m all around, its edge at most 80 m behind it and 100 m without forest behind the edge
 * - forestFlightKeepsItsClearances: the flight into the forest starts at the start, keeps 0.5 m above the soil,
 *   0.7 m from every stem and 0.35 m from every leaf of the trees near its way at 30 frames per second of 24 s, and
 * stands 1.7 m above the soil at the end from u = 0.85 on
 * - forestFlightIsSmooth: from frame to frame of those 24 s at 30 fps the velocity changes by at most a fifth of
 *   the largest speed within half a second (or 0.1 m per frame), and the view turns by at most 3.5 degrees
 **/

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "klartraum/tree/bonsai_flight.hpp"
#include "klartraum/tree/tree_placement.hpp"

using namespace klartraum;

namespace {

// The reference flight of tree_bonsai_example --fly-in: from the orbit at distance 4.2, elevation 0.32, azimuth
// -2.09 to the ground place (-1.04, 0.15), 1.7 m above it, heading 120 deg.
struct ReferenceFlight {
    BonsaiScene scene;
    glm::vec3 start;
    glm::vec2 ground{-1.04f, 0.15f};
    float above;
    float heading = glm::radians(120.0f);

    ReferenceFlight() {
        BonsaiSceneConfig config;
        config.scale.s = 1.0f;
        scene = makeBonsaiScene(config);
        const float azimuth = -2.09f, elevation = 0.32f, distance = 4.2f;
        start = scene.focus + distance * glm::vec3(std::sin(azimuth) * std::cos(elevation), std::sin(elevation),
                                                   std::cos(azimuth) * std::cos(elevation));
        above = 1.7f / scene.config.scale.metersPerUnit();
    }

    FlyInPose at(float u) const { return flyInPose(scene, start, scene.focus, ground, above, heading, u); }
};

} // namespace

TEST(BonsaiFlightTest, smootherstepEasesInAndOut) {
    EXPECT_EQ(smootherstep(-1.0f), 0.0f);
    EXPECT_EQ(smootherstep(2.0f), 1.0f);
    EXPECT_NEAR(smootherstep(0.5f), 0.5f, 1e-6f);
    EXPECT_LT(smootherstep(0.01f), 1e-5f);
    EXPECT_GT(smootherstep(0.99f), 1.0f - 1e-5f);
}

TEST(BonsaiFlightTest, slerpDirectionStaysOnTheSphere) {
    const glm::vec3 a(1.0f, 0.0f, 0.0f), b = glm::normalize(glm::vec3(0.0f, 1.0f, 1.0f));
    for (float t = 0.0f; t <= 1.0f; t += 0.125f) {
        EXPECT_NEAR(glm::length(slerpDirection(a, b, t)), 1.0f, 1e-5f) << t;
    }
    EXPECT_LT(glm::length(slerpDirection(a, b, 0.0f) - a), 1e-5f);
    EXPECT_LT(glm::length(slerpDirection(a, b, 1.0f) - b), 1e-5f);
    EXPECT_EQ(slerpDirection(b, b, 0.3f), b);
}

TEST(BonsaiFlightTest, flightStartsAtTheStart) {
    const ReferenceFlight flight;
    const FlyInPose pose = flight.at(0.0f);
    EXPECT_LT(glm::length(pose.eye - flight.start), 1e-4f);
    const glm::vec3 look = glm::normalize(pose.lookAt - pose.eye);
    EXPECT_GT(glm::dot(look, glm::normalize(flight.scene.focus - flight.start)), 0.9999f);
    EXPECT_FLOAT_EQ(pose.nearPlane, 0.05f);
}

TEST(BonsaiFlightTest, flightEndsAboveTheGround) {
    const ReferenceFlight flight;
    for (int i = 0; i <= 480; ++i) {
        const FlyInPose pose = flight.at(float(i) / 480.0f);
        const float soil = flight.scene.soilHeight(pose.eye.x, pose.eye.z);
        ASSERT_GE(pose.eye.y, soil + flight.above - 1e-6f) << i;
        ASSERT_GT(pose.nearPlane, 0.0f) << i;
    }
    const FlyInPose end = flight.at(1.0f);
    const float soil = flight.scene.soilHeight(end.eye.x, end.eye.z);
    EXPECT_LT(end.eye.y - soil, 2.0f * flight.above);
    EXPECT_LT(glm::length(glm::vec2(end.eye.x, end.eye.z) - flight.ground), 4.0f * flight.above);
    const float h = flight.heading + glm::radians(35.0f);
    const glm::vec3 look = glm::normalize(end.lookAt - end.eye);
    EXPECT_GT(glm::dot(glm::normalize(glm::vec2(look.x, look.z)), glm::vec2(std::sin(h), -std::cos(h))), 0.999f);
}

namespace {

const ForestFlight& forestFlight() {
    static const ReferenceFlight flight;
    static const ForestFlight forest(flight.scene, flight.start, flight.scene.focus, findForestApproach(flight.scene));
    return forest;
}

} // namespace

TEST(BonsaiFlightTest, forestApproachLiesInDenseForest) {
    const ReferenceFlight flight;
    const BonsaiScene& scene = flight.scene;
    const std::vector<ForestApproach> approaches = findForestApproach(scene);
    ASSERT_FALSE(approaches.empty());
    const ForestApproach& best = approaches.front();
    const float m = 1.0f / scene.config.scale.metersPerUnit();
    std::printf("%zu approaches; the best at (%.4f, %.4f), heading %.0f deg, edge %.0f m behind\n", approaches.size(),
                best.end.x, best.end.y, glm::degrees(best.heading), best.edgeMeters);
    for (float r = 0.0f; r <= 30.0f; r += 5.0f) {
        for (int k = 0; k < 16; ++k) {
            const float a = 2.0f * 3.14159265f * float(k) / 16.0f;
            const glm::vec2 p = best.end + r * m * glm::vec2(std::cos(a), std::sin(a));
            EXPECT_GE(tree::treeLandCover(scene.terrain, p.x, p.y).vegetation.x, 0.9f) << r << " " << k;
        }
    }
    EXPECT_LE(best.edgeMeters, 80.0f);
    const glm::vec2 back = -glm::vec2(std::sin(best.heading), -std::cos(best.heading));
    for (float d = best.edgeMeters + 5.0f; d <= best.edgeMeters + 100.0f; d += 5.0f) {
        const glm::vec2 p = best.end + d * m * back;
        const glm::vec4 cover = tree::treeLandCover(scene.terrain, p.x, p.y).vegetation;
        EXPECT_LT(cover.x, 0.2f) << d;
    }
}

TEST(BonsaiFlightTest, forestFlightKeepsItsClearances) {
    const ReferenceFlight flight;
    const ForestFlight& forest = forestFlight();
    ASSERT_TRUE(forest.isValid());
    const BonsaiScene& scene = flight.scene;
    const float m = 1.0f / scene.config.scale.metersPerUnit();
    EXPECT_LT(glm::length(forest.at(0.0f).eye - flight.start), 1e-4f);
    float soil = 1e9f, trunk = 1e9f;
    for (int i = 0; i < 720; ++i) {
        const float u = float(i) / 719.0f;
        const glm::vec3 eye = forest.at(u).eye;
        soil = std::min(soil, (eye.y - scene.soilHeight(eye.x, eye.z)) / m);
        trunk = std::min(trunk, forest.clearanceAt(eye));
        if (u >= 0.85f) {
            EXPECT_NEAR((eye.y - scene.soilHeight(eye.x, eye.z)) / m, 1.7f, 0.05f) << i;
            EXPECT_LT(glm::length(eye - forest.at(1.0f).eye), 0.01f * m) << i;
        }
    }
    std::printf("%zu waypoints; at least %.2f m above the soil and %.2f m from the trees\n",
                forest.getWaypoints().size(), soil, trunk);
    EXPECT_GE(soil, 0.5f);
    // clearanceAt() widens the stems by 0.7 - 0.35 m.
    EXPECT_GE(trunk, 0.35f);
}

TEST(BonsaiFlightTest, forestFlightIsSmooth) {
    const ReferenceFlight flight;
    const ForestFlight& forest = forestFlight();
    ASSERT_TRUE(forest.isValid());
    const float m = 1.0f / flight.scene.config.scale.metersPerUnit();
    std::vector<float> speed(720, 0.0f);
    for (int i = 1; i < 720; ++i) {
        speed[i] = glm::length(forest.at(float(i) / 719.0f).eye - forest.at(float(i - 1) / 719.0f).eye);
    }
    float worstAcceleration = 0.0f, worstTurn = 0.0f;
    int worstFrame = 0;
    for (int i = 1; i + 1 < 720; ++i) {
        float nearby = 0.0f;
        for (int j = std::max(i - 15, 0); j <= std::min(i + 15, 719); ++j) {
            nearby = std::max(nearby, speed[j]);
        }
        const FlyInPose a = forest.at(float(i - 1) / 719.0f), b = forest.at(float(i) / 719.0f),
                        c = forest.at(float(i + 1) / 719.0f);
        const glm::vec3 v0 = b.eye - a.eye, v1 = c.eye - b.eye;
        const float change = glm::length(v1 - v0) / std::max(0.2f * nearby, 0.1f * m);
        if (change > worstAcceleration) {
            worstAcceleration = change;
            worstFrame = i;
        }
        const float turn = glm::degrees(std::acos(
            std::clamp(glm::dot(glm::normalize(b.lookAt - b.eye), glm::normalize(c.lookAt - c.eye)), -1.0f, 1.0f)));
        worstTurn = std::max(worstTurn, turn);
    }
    std::printf("velocity changes by at most %.2f of the bound (frame %d); the view turns by at most %.2f deg\n",
                worstAcceleration, worstFrame, worstTurn);
    EXPECT_LE(worstAcceleration, 1.0f);
    EXPECT_LE(worstTurn, 3.5f);
}
