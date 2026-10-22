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
 **/

#include <gtest/gtest.h>

#include <cmath>

#include "klartraum/tree/bonsai_flight.hpp"

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
