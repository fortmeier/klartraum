// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_SCENE_PLACEMENT_HPP
#define KLARTRAUM_SCENE_PLACEMENT_HPP

#include <vector>

#include <glm/glm.hpp>

#include "klartraum/character_controller.hpp"
#include "klartraum/rendering/gaussian_data_standard.hpp"

namespace klartraum {

/**
 * @brief Where an upright object stands in a Gaussian scene, measured from its Gaussians (y up).
 */
struct UprightObjectBounds {
    glm::vec3 base;        ///< centre of the object's footprint on the ground
    float height;          ///< from the ground to the top of the object
    float footprintRadius; ///< horizontal radius of the object above the ground
    float groundRadius;    ///< horizontal radius of the ground around the object, e.g. a patch of moss
};

/**
 * @brief Measures the object standing in the middle of a scene, e.g. the lantern on its patch of moss.
 *
 * The object is the column of Gaussians around the horizontal median of all
 * Gaussians whose opacity is at least `minimumAlpha`. The ground is the low
 * percentile of the heights of the Gaussians around the column (between
 * `columnRadius` and three times it), the top the high percentile of the
 * heights inside the column, and the footprint radius the high percentile of
 * the distances of the Gaussians that rise clearly above the ground. The ground
 * radius is the distance that 80 % of the Gaussians near ground level lie
 * within. Percentiles instead of extremes ignore stray Gaussians.
 */
UprightObjectBounds measureUprightObject(const std::vector<Gaussian3D>& gaussians, float columnRadius,
                                         float minimumAlpha = 0.5f);

/**
 * @brief Places a character in a scene: maps its world space (metres, y up, ground y = 0) into scene space.
 *
 * The character's origin goes to `base`, and it is scaled so that a character
 * of `characterHeight` metres is `sceneHeight` scene units tall.
 */
glm::mat4 characterToScene(glm::vec3 base, float sceneHeight, float characterHeight);

/**
 * @brief Steers a CharacterController along a circle on the ground.
 *
 * Each control faces along the circle (counter-clockwise seen from above, or
 * clockwise) and corrects towards it: the desired direction is the tangent
 * tilted towards the circle in proportion to the distance from it, up to 45
 * degrees. The angle around the centre is accumulated, so completedTurns()
 * tells how often the character went around.
 */
class CirclePathFollower {
public:
    CirclePathFollower(glm::vec2 centre, float radius, float speed, bool clockwise = false, uint32_t style = 0);

    /// The control for a character at `position` (ground x, z of the hips); updates the accumulated angle.
    CharacterControl control(glm::vec2 position);

    /// Turns around the centre completed so far (fractional).
    float completedTurns() const;

private:
    glm::vec2 centre;
    float radius;
    float speed;
    bool clockwise;
    uint32_t style;
    bool started = false;
    float previousAngle = 0.0f;
    float accumulatedAngle = 0.0f;
};

/**
 * @brief Steers a CharacterController into a circle from the side: straight ahead, then around.
 *
 * The character starts at the origin facing +z (as CharacterController's
 * initial state does) and walks straight along +z for `approach` metres, held on
 * that line. There it meets the circle of `radius` around (-radius, approach)
 * along its tangent and continues counter-clockwise around it with a
 * CirclePathFollower. With an `approachSpeed` (e.g. running), the approach
 * is taken at that speed, and on the circle the speed eases to `speed` over
 * `slowDownSeconds`, counted in controls of `stepSeconds` each.
 */
class WalkInPathFollower {
public:
    WalkInPathFollower(float approach, float radius, float speed, uint32_t style = 0, float approachSpeed = 0.0f,
                       float slowDownSeconds = 1.5f, float stepSeconds = 1.0f / 30.0f);

    /// The control for a character at `position` (ground x, z of the hips).
    CharacterControl control(glm::vec2 position);

    /// Whether the character has reached the circle.
    bool onCircle() const { return circleReached; }
    /// Turns around the circle since reaching it.
    float completedTurns() const { return circle.completedTurns(); }

private:
    float approach;
    float speed;
    uint32_t style;
    float approachSpeed;
    float slowDownSeconds;
    float stepSeconds;
    uint32_t stepsOnCircle = 0;
    bool circleReached = false;
    CirclePathFollower circle;
};

/**
 * @brief Places a WalkInPathFollower's character world in a scene so the character walks in from the side.
 *
 * The circle of `pathRadius` (scene units) lies around `base`; the character joins it at the point nearest
 * `camera` (seen from above) and starts `approach` scene units before it, at right angles to the camera's
 * direction, so it crosses the view sideways. Scaled like characterToScene().
 */
glm::mat4 walkInToScene(glm::vec3 base, float pathRadius, glm::vec3 camera, float approach, float sceneHeight,
                        float characterHeight);

} // namespace klartraum

#endif // KLARTRAUM_SCENE_PLACEMENT_HPP
