// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - measuresSyntheticColumn: a column of Gaussians standing on a disc of ground Gaussians is measured with its base on
 *   the ground below the column, its height and its radius, and the ground with the radius that 80 % of its Gaussians
 *   lie within
 * - ignoresTransparentGaussians: Gaussians below the opacity threshold do not count, even a tall transparent spike
 * - characterToSceneScalesToHeight: the character's origin goes to the base, and its head height to the scene height
 *   above it
 * - followerSteersAlongTangent: on the circle, the control faces along it, counter-clockwise or clockwise, at the
 *   requested speed
 * - followerCorrectsTowardsCircle: outside the circle the control turns inwards, inside it outwards
 * - followerCountsTurns: positions going once around the centre in small steps count as one completed turn
 * - walkInGoesStraightThenAround: following the walk-in control from the origin, a character walks straight
 *   along +z to the circle, joins it along its tangent without a turn, then circles it counter-clockwise, staying on it
 * - walkInRunsThenSlowsToWalk: with a faster approach speed, the character runs the straight approach and, once on
 *   the circle, slows down steadily to the walking speed within the slow-down time
 * - walkInToScenePlacesTheApproach: in the scene, the character's circle is centred on the base, it joins the
 *   circle at the point nearest the camera, and it starts the approach distance beside it, at right angles to the
 *   camera's direction
 * - controllerWalksAroundCircle: the character controller steered by the follower walks a full circle of 1.5 m
 *   radius, staying within 35 cm of it once it has reached it
 * - lanternIsMeasured: the lantern of data/lantern.spz stands near the middle of its scene, upright, taller than
 *   wide
 **/

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <filesystem>
#include <memory>
#include <vector>

#include <glm/glm.hpp>

#include "klartraum/character_controller.hpp"
#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/gaussian_data_standard.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/scene_placement.hpp"

using namespace klartraum;

namespace {

constexpr float kPi = 3.14159265358979323846f;

Gaussian3D gaussianAt(glm::vec3 position, float alpha = 1.0f) {
    Gaussian3D gaussian{};
    gaussian.position = {position.x, position.y, position.z};
    gaussian.alpha = alpha;
    return gaussian;
}

// A disc of ground at y = 1 (radius 0.3) around (2, 3) and a column of radius 0.05 and height 0.2 on it.
std::vector<Gaussian3D> columnOnGround() {
    std::vector<Gaussian3D> gaussians;
    for (int ring = 1; ring <= 30; ++ring) {
        for (int step = 0; step < 60; ++step) {
            const float angle = 2.0f * kPi * float(step) / 60.0f, radius = 0.01f * float(ring);
            gaussians.push_back(gaussianAt({2.0f + radius * std::cos(angle), 1.0f, 3.0f + radius * std::sin(angle)}));
        }
    }
    for (int level = 0; level <= 40; ++level) {
        for (int step = 0; step < 24; ++step) {
            const float angle = 2.0f * kPi * float(step) / 24.0f;
            gaussians.push_back(gaussianAt(
                {2.0f + 0.05f * std::cos(angle), 1.0f + 0.005f * float(level), 3.0f + 0.05f * std::sin(angle)}));
        }
    }
    return gaussians;
}

float angleBetween(glm::vec2 a, glm::vec2 b) {
    return std::acos(std::clamp(glm::dot(glm::normalize(a), glm::normalize(b)), -1.0f, 1.0f));
}

} // namespace

TEST(ScenePlacement, measuresSyntheticColumn) {
    const auto bounds = measureUprightObject(columnOnGround(), 0.08f);
    EXPECT_NEAR(bounds.base.x, 2.0f, 0.01f);
    EXPECT_NEAR(bounds.base.y, 1.0f, 1e-4f);
    EXPECT_NEAR(bounds.base.z, 3.0f, 0.01f);
    EXPECT_NEAR(bounds.height, 0.2f, 0.005f);
    EXPECT_NEAR(bounds.footprintRadius, 0.05f, 0.005f);
    // Equally many Gaussians on every ring of radius 0.01 to 0.3: 80 % lie within 0.24.
    EXPECT_NEAR(bounds.groundRadius, 0.24f, 0.011f);
}

TEST(ScenePlacement, ignoresTransparentGaussians) {
    auto gaussians = columnOnGround();
    for (int level = 0; level < 2000; ++level) {
        gaussians.push_back(gaussianAt({2.0f, 1.0f + 0.001f * float(level), 3.0f}, 0.2f));
    }
    EXPECT_NEAR(measureUprightObject(gaussians, 0.08f).height, 0.2f, 0.005f);
}

TEST(ScenePlacement, characterToSceneScalesToHeight) {
    const glm::mat4 transform = characterToScene({1.0f, 2.0f, 3.0f}, 0.1f, 1.7f);
    const glm::vec4 origin = transform * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    const glm::vec4 head = transform * glm::vec4(0.0f, 1.7f, 0.0f, 1.0f);
    EXPECT_NEAR(glm::length(glm::vec3(origin) - glm::vec3(1.0f, 2.0f, 3.0f)), 0.0f, 1e-6f);
    EXPECT_NEAR(glm::length(glm::vec3(head) - glm::vec3(1.0f, 2.1f, 3.0f)), 0.0f, 1e-6f);
}

TEST(ScenePlacement, followerSteersAlongTangent) {
    CirclePathFollower counterClockwise({1.0f, 1.0f}, 2.0f, 1.3f);
    const auto control = counterClockwise.control({3.0f, 1.0f});
    EXPECT_NEAR(glm::length(control.velocity), 1.3f, 1e-5f);
    EXPECT_LT(angleBetween(control.facing, {0.0f, 1.0f}), 1e-4f);
    EXPECT_LT(angleBetween(control.velocity, control.facing), 1e-4f);

    CirclePathFollower clockwise({1.0f, 1.0f}, 2.0f, 1.3f, true);
    EXPECT_LT(angleBetween(clockwise.control({3.0f, 1.0f}).facing, {0.0f, -1.0f}), 1e-4f);
}

TEST(ScenePlacement, followerCorrectsTowardsCircle) {
    CirclePathFollower follower({0.0f, 0.0f}, 2.0f, 1.0f);
    // Outside at (3, 0): the direction points inwards (negative x) while still going around (positive z).
    const auto outside = follower.control({3.0f, 0.0f});
    EXPECT_LT(outside.facing.x, -0.1f);
    EXPECT_GT(outside.facing.y, 0.1f);
    // Far outside: at most 45 degrees off the tangent.
    EXPECT_NEAR(angleBetween(follower.control({20.0f, 0.0f}).facing, {0.0f, 1.0f}), kPi / 4.0f, 1e-4f);
    const auto inside = follower.control({1.0f, 0.0f});
    EXPECT_GT(inside.facing.x, 0.1f);
}

TEST(ScenePlacement, followerCountsTurns) {
    CirclePathFollower follower({0.5f, -0.5f}, 1.0f, 1.0f);
    for (int step = 0; step <= 100; ++step) {
        const float angle = 2.0f * kPi * float(step) / 100.0f;
        follower.control({0.5f + std::cos(angle), -0.5f + std::sin(angle)});
    }
    EXPECT_NEAR(follower.completedTurns(), 1.0f, 1e-4f);
}

TEST(ScenePlacement, walkInGoesStraightThenAround) {
    const float approach = 4.0f, radius = 1.5f, speed = 1.2f, dt = 1.0f / 30.0f;
    WalkInPathFollower follower(approach, radius, speed);
    glm::vec2 position(0.0f);
    float previousHeading = 0.0f;
    bool joined = false;
    int stepsOnCircle = 0;
    for (int step = 0; step < 30 * 20; ++step) {
        const CharacterControl control = follower.control(position);
        EXPECT_NEAR(glm::length(control.velocity), speed, 1e-4f) << step;
        if (!follower.onCircle()) {
            EXPECT_NEAR(position.x, 0.0f, 1e-5f) << step;
            EXPECT_NEAR(control.facing.x, 0.0f, 1e-5f) << step;
        } else {
            if (!joined) {
                // The first control on the circle still points along +z: no turn at the join.
                EXPECT_NEAR(control.facing.x, 0.0f, 0.05f);
                EXPECT_GT(control.facing.y, 0.99f);
                joined = true;
            }
            ++stepsOnCircle;
            const float distance = std::abs(glm::distance(position, glm::vec2(-radius, approach)) - radius);
            EXPECT_LT(distance, 0.05f) << step;
        }
        const float heading = std::atan2(control.facing.x, control.facing.y);
        // The heading's change, wrapped to [-pi, pi]: no sudden turn.
        const float turn = std::remainder(heading - previousHeading, 6.2831853f);
        EXPECT_LT(std::abs(turn), 0.1f) << step;
        previousHeading = heading;
        position += control.velocity * dt;
    }
    EXPECT_TRUE(joined);
    EXPECT_GT(follower.completedTurns(), 1.0f);
}

TEST(ScenePlacement, walkInRunsThenSlowsToWalk) {
    const float approach = 6.0f, radius = 1.5f, walk = 1.2f, run = 3.5f, slowDown = 1.5f, dt = 1.0f / 30.0f;
    WalkInPathFollower follower(approach, radius, walk, 0, run, slowDown, dt);
    glm::vec2 position(0.0f);
    float previousSpeed = run;
    int stepsOnCircle = 0;
    for (int step = 0; step < 30 * 10; ++step) {
        const CharacterControl control = follower.control(position);
        const float speed = glm::length(control.velocity);
        if (!follower.onCircle()) {
            EXPECT_NEAR(speed, run, 1e-4f) << step;
        } else {
            EXPECT_LE(speed, previousSpeed + 1e-4f) << step;
            EXPECT_GE(speed, walk - 1e-4f) << step;
            if (float(stepsOnCircle) * dt >= slowDown) {
                EXPECT_NEAR(speed, walk, 1e-4f) << step;
            }
            ++stepsOnCircle;
        }
        previousSpeed = speed;
        position += control.velocity * dt;
    }
    EXPECT_GT(stepsOnCircle, 0);
}

TEST(ScenePlacement, walkInToScenePlacesTheApproach) {
    const glm::vec3 base(0.1f, -0.2f, 0.3f), camera(2.0f, 1.0f, 1.0f);
    const float pathRadius = 0.2f, approach = 0.5f, sceneHeight = 0.15f, characterHeight = 1.75f;
    const glm::mat4 toScene = walkInToScene(base, pathRadius, camera, approach, sceneHeight, characterHeight);
    const float scale = sceneHeight / characterHeight;
    const float radius = pathRadius / scale, length = approach / scale;
    auto map = [&](glm::vec2 p) { return glm::vec3(toScene * glm::vec4(p.x, 0.0f, p.y, 1.0f)); };

    const glm::vec3 centre = map({-radius, length});
    EXPECT_NEAR(glm::distance(centre, base), 0.0f, 1e-5f);
    const glm::vec3 entry = map({0.0f, length});
    const glm::vec3 towardsCamera = glm::normalize(glm::vec3(camera.x - base.x, 0.0f, camera.z - base.z));
    EXPECT_NEAR(glm::distance(entry, base + pathRadius * towardsCamera), 0.0f, 1e-5f);
    const glm::vec3 start = map({0.0f, 0.0f});
    EXPECT_NEAR(glm::distance(start, entry), approach, 1e-5f);
    EXPECT_NEAR(glm::dot(glm::normalize(entry - start), towardsCamera), 0.0f, 1e-5f);
    EXPECT_NEAR(start.y, base.y, 1e-6f);
    // One metre up in the character's world is `scale` up in the scene.
    EXPECT_NEAR(glm::vec3(toScene * glm::vec4(0.0f, 1.0f, 0.0f, 1.0f)).y - start.y, scale, 1e-6f);
}

TEST(ScenePlacement, controllerWalksAroundCircle) {
    if (!std::filesystem::exists("data/character/controller.onnx")) {
        GTEST_SKIP() << "Missing data/character/controller.onnx";
    }
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    auto controller = std::make_shared<CharacterController>(vc, "data/character");
    ComputeGraph graph(vc, 1);
    graph.compileFrom(controller);

    // The character starts at the origin facing +z; the circle passes through its start, centred to its left.
    const float radius = 1.5f;
    CirclePathFollower follower({radius, 0.0f}, radius, 1.2f, true);
    float largestDeviation = 0.0f;
    uint32_t steps = 0;
    while (follower.completedTurns() < 1.0f && steps < 2000) {
        const glm::vec4 hips = controller->readJoints()[0];
        const glm::vec2 position(hips.x, hips.z);
        controller->setControl(follower.control(position));
        graph.submitAndWait(vc.getGraphicsQueue(), 0);
        if (steps > 60) {
            largestDeviation =
                std::max(largestDeviation, std::abs(glm::length(position - glm::vec2(radius, 0.0f)) - radius));
        }
        ++steps;
    }
    EXPECT_GE(follower.completedTurns(), 1.0f) << "after " << steps << " steps";
    // About 9.4 m at 1.2 m/s and 30 steps per second, with a margin for starting and turning.
    EXPECT_LT(steps, 450u);
    EXPECT_LT(largestDeviation, 0.35f);
}

TEST(ScenePlacement, lanternIsMeasured) {
    if (!std::filesystem::exists("data/lantern.spz")) {
        GTEST_SKIP() << "Missing data/lantern.spz";
    }
    const auto gaussians = loadGaussiansSpz("data/lantern.spz", true);
    const auto bounds = measureUprightObject(gaussians, 0.04f);
    std::cout << "lantern base " << bounds.base.x << " " << bounds.base.y << " " << bounds.base.z << ", height "
              << bounds.height << ", footprint radius " << bounds.footprintRadius << ", ground radius "
              << bounds.groundRadius << std::endl;
    EXPECT_GT(bounds.height, 0.0f);
    EXPECT_GT(bounds.height, bounds.footprintRadius);
    EXPECT_GT(bounds.groundRadius, bounds.footprintRadius);
    EXPECT_LT(glm::length(glm::vec2(bounds.base.x, bounds.base.z)), 0.5f * bounds.height);
}
