// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_TREE_BONSAI_FLIGHT_HPP
#define KLARTRAUM_TREE_BONSAI_FLIGHT_HPP

#include <glm/glm.hpp>

#include "klartraum/tree/bonsai_scene.hpp"

namespace klartraum {

/// 6x^5 - 15x^4 + 10x^3 of x clamped to [0, 1].
float smootherstep(float x);

/// The spherical interpolation from the unit direction `a` to `b` at t.
glm::vec3 slerpDirection(glm::vec3 a, glm::vec3 b, float t);

struct FlyInPose {
    glm::vec3 eye;
    glm::vec3 lookAt;
    float nearPlane;
};

/// The flight from `start` (looking at `focus`) down to `ground`, `above` units above the soil and looking along
/// `heading` (radians from -z towards +x), at u in [0, 1]. The camera's distance to the place shrinks
/// geometrically, so each halving of the distance takes the same time, while its direction from the place turns
/// from the start's to behind the heading. The camera looks at the place and, in the last quarter, turns to look
/// ahead; during the last quarter it pans slowly around. It stays above the soil, and the near plane follows its
/// height above it.
FlyInPose flyInPose(const BonsaiScene& scene, glm::vec3 start, glm::vec3 focus, glm::vec2 ground, float above,
                    float heading, float u);

} // namespace klartraum

#endif // KLARTRAUM_TREE_BONSAI_FLIGHT_HPP
