// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <glm/gtc/matrix_transform.hpp>

#include "klartraum/scene_placement.hpp"

namespace klartraum {

namespace {

constexpr float kPi = 3.14159265358979323846f;

float percentile(std::vector<float> values, float fraction) {
    if (values.empty()) {
        throw std::runtime_error("measureUprightObject: no Gaussians in the measured region");
    }
    const auto index = static_cast<size_t>(std::clamp(fraction, 0.0f, 1.0f) * float(values.size() - 1));
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(index), values.end());
    return values[index];
}

} // namespace

UprightObjectBounds measureUprightObject(const std::vector<Gaussian3D>& gaussians, float columnRadius,
                                         float minimumAlpha) {
    std::vector<glm::vec3> positions;
    for (const auto& gaussian : gaussians) {
        if (gaussian.alpha >= minimumAlpha) {
            positions.emplace_back(gaussian.position[0], gaussian.position[1], gaussian.position[2]);
        }
    }
    std::vector<float> xs, zs;
    for (const auto& position : positions) {
        xs.push_back(position.x);
        zs.push_back(position.z);
    }
    const glm::vec2 centre(percentile(xs, 0.5f), percentile(zs, 0.5f));

    std::vector<float> groundHeights, columnHeights;
    for (const auto& position : positions) {
        const float distance = glm::length(glm::vec2(position.x, position.z) - centre);
        if (distance < columnRadius) {
            columnHeights.push_back(position.y);
        } else if (distance < 3.0f * columnRadius) {
            groundHeights.push_back(position.y);
        }
    }
    const float ground = percentile(groundHeights, 0.1f);
    const float top = percentile(columnHeights, 0.995f);

    // The object's own Gaussians rise clearly above the ground; moss and stones do not.
    const float clearance = ground + 0.15f * (top - ground);
    std::vector<float> distances;
    for (const auto& position : positions) {
        const float distance = glm::length(glm::vec2(position.x, position.z) - centre);
        if (position.y > clearance && distance < 3.0f * columnRadius) {
            distances.push_back(distance);
        }
    }
    std::vector<float> groundDistances;
    for (const auto& position : positions) {
        if (position.y < ground + 0.1f * (top - ground)) {
            groundDistances.push_back(glm::length(glm::vec2(position.x, position.z) - centre));
        }
    }
    return {glm::vec3(centre.x, ground, centre.y), top - ground, percentile(distances, 0.98f),
            percentile(groundDistances, 0.8f)};
}

glm::mat4 characterToScene(glm::vec3 base, float sceneHeight, float characterHeight) {
    if (!(characterHeight > 0.0f)) {
        throw std::runtime_error("characterToScene: the character height must be positive");
    }
    return glm::scale(glm::translate(glm::mat4(1.0f), base), glm::vec3(sceneHeight / characterHeight));
}

CirclePathFollower::CirclePathFollower(glm::vec2 centre, float radius, float speed, bool clockwise, uint32_t style)
    : centre(centre),
      radius(radius),
      speed(speed),
      clockwise(clockwise),
      style(style) {
    if (!(radius > 0.0f)) {
        throw std::runtime_error("CirclePathFollower: the radius must be positive");
    }
}

CharacterControl CirclePathFollower::control(glm::vec2 position) {
    glm::vec2 outward = position - centre;
    const float distance = glm::length(outward);
    outward = distance > 1e-6f ? outward / distance : glm::vec2(1.0f, 0.0f);

    const float angle = std::atan2(outward.y, outward.x);
    if (started) {
        float delta = angle - previousAngle;
        delta -= 2.0f * kPi * std::round(delta / (2.0f * kPi));
        accumulatedAngle += delta;
    }
    started = true;
    previousAngle = angle;

    // Counter-clockwise in the (x, z) plane; z points towards the viewer when looking down -y.
    const glm::vec2 tangent = clockwise ? glm::vec2(outward.y, -outward.x) : glm::vec2(-outward.y, outward.x);
    // Too far out: tilt inwards; too close: tilt outwards; at most 45 degrees.
    const float correction = std::clamp((distance - radius) / radius * 2.0f, -1.0f, 1.0f);
    const glm::vec2 direction = glm::normalize(tangent - correction * outward);

    CharacterControl result;
    result.facing = direction;
    result.velocity = direction * speed;
    result.style = style;
    return result;
}

float CirclePathFollower::completedTurns() const { return std::abs(accumulatedAngle) / (2.0f * kPi); }

WalkInPathFollower::WalkInPathFollower(float approach, float radius, float speed, uint32_t style, float approachSpeed,
                                       float slowDownSeconds, float stepSeconds)
    : approach(approach),
      speed(speed),
      style(style),
      approachSpeed(approachSpeed > 0.0f ? approachSpeed : speed),
      slowDownSeconds(slowDownSeconds),
      stepSeconds(stepSeconds),
      circle({-radius, approach}, radius, speed, false, style) {}

CharacterControl WalkInPathFollower::control(glm::vec2 position) {
    if (!circleReached && position.y >= approach) {
        circleReached = true;
    }
    if (circleReached) {
        CharacterControl result = circle.control(position);
        // From the approach speed to the circle's speed, eased in and out.
        const float t = slowDownSeconds > 0.0f
                            ? std::clamp(float(stepsOnCircle) * stepSeconds / slowDownSeconds, 0.0f, 1.0f)
                            : 1.0f;
        const float eased = t * t * (3.0f - 2.0f * t);
        result.velocity *= (approachSpeed + (speed - approachSpeed) * eased) / speed;
        ++stepsOnCircle;
        return result;
    }
    // Straight along +z, turned back towards the line x = 0 in proportion to
    // the distance from it, as the circle follower does with its circle.
    const float tilt = std::clamp(-position.x, -1.0f, 1.0f) * 0.785398f;
    const glm::vec2 direction(std::sin(tilt), std::cos(tilt));
    CharacterControl result;
    result.velocity = direction * approachSpeed;
    result.facing = direction;
    result.style = style;
    return result;
}

glm::mat4 walkInToScene(glm::vec3 base, float pathRadius, glm::vec3 camera, float approach, float sceneHeight,
                        float characterHeight) {
    if (!(characterHeight > 0.0f) || !(pathRadius > 0.0f)) {
        throw std::runtime_error("walkInToScene: the character height and the path radius must be positive");
    }
    glm::vec2 towards(camera.x - base.x, camera.z - base.z);
    if (glm::length(towards) < 1e-9f) {
        towards = glm::vec2(1.0f, 0.0f);
    }
    towards = glm::normalize(towards);
    // The character's +x (away from its circle's centre, which is on its
    // left) points towards the camera; its +z, the direction it walks in,
    // is at right angles to that: a turn by theta about y.
    const float theta = std::atan2(-towards.y, towards.x);
    const glm::vec3 heading(std::sin(theta), 0.0f, std::cos(theta));
    const glm::vec3 entry = base + pathRadius * glm::vec3(towards.x, 0.0f, towards.y);
    const glm::vec3 start = entry - approach * heading;
    const glm::mat4 turn = glm::rotate(glm::mat4(1.0f), theta, glm::vec3(0.0f, 1.0f, 0.0f));
    return glm::scale(glm::translate(glm::mat4(1.0f), start) * turn, glm::vec3(sceneHeight / characterHeight));
}

} // namespace klartraum
