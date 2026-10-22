// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/tree/bonsai_flight.hpp"

#include <algorithm>
#include <cmath>

namespace klartraum {

float smootherstep(float x) {
    x = std::clamp(x, 0.0f, 1.0f);
    return x * x * x * (x * (6.0f * x - 15.0f) + 10.0f);
}

glm::vec3 slerpDirection(glm::vec3 a, glm::vec3 b, float t) {
    const float angle = std::acos(std::clamp(glm::dot(a, b), -1.0f, 1.0f));
    if (angle < 1e-4f) {
        return b;
    }
    return (std::sin((1.0f - t) * angle) * a + std::sin(t * angle) * b) / std::sin(angle);
}

FlyInPose flyInPose(const BonsaiScene& scene, glm::vec3 start, glm::vec3 focus, glm::vec2 ground, float above,
                    float heading, float u) {
    const float x = std::clamp(u / 0.8f, 0.0f, 1.0f);
    const float descent = x * x * (3.0f - 2.0f * x);
    const float pan = smootherstep((u - 0.75f) / 0.25f);
    const glm::vec3 place(ground.x, scene.soilHeight(ground.x, ground.y) + above, ground.y);
    const float h = heading + glm::radians(35.0f) * pan;
    const glm::vec3 forward(std::sin(h), 0.0f, -std::cos(h));
    const glm::vec3 behind = glm::normalize(-forward + glm::vec3(0.0f, 0.25f, 0.0f));
    const float startDistance = glm::length(start - place);
    const float endDistance = 2.0f * above;
    const float d = startDistance * std::pow(endDistance / startDistance, descent);
    const glm::vec3 dir = slerpDirection(glm::normalize(start - place), behind, descent);
    glm::vec3 eye = place + d * dir;
    const float soil = scene.soilHeight(eye.x, eye.z);
    eye.y = std::max(eye.y, soil + above);

    const float ahead = smootherstep((descent - 0.75f) / 0.25f);
    const glm::vec3 toPlace = glm::normalize(glm::mix(focus, place, smootherstep(descent / 0.3f)) - eye);
    const glm::vec3 lookAhead = glm::normalize(forward + glm::vec3(0.0f, -0.08f, 0.0f));
    const glm::vec3 look = slerpDirection(toPlace, lookAhead, ahead);
    return {eye, eye + look, std::clamp(0.5f * (eye.y - soil), 1e-5f, 0.05f)};
}

} // namespace klartraum
