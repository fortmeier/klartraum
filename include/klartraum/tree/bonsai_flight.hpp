// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_TREE_BONSAI_FLIGHT_HPP
#define KLARTRAUM_TREE_BONSAI_FLIGHT_HPP

#include <vector>

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

/// What findForestApproach() looks for and how ForestFlight flies there, in metres of the landscape.
struct ForestApproachConfig {
    float forestRadius = 30.0f; ///< forest (weight >= minForest) all around the end, this far
    float minForest = 0.9f;
    float maxEdge = 80.0f;       ///< the forest's edge at most this far behind the end
    float fieldLength = 100.0f;  ///< no forest this far behind the edge
    float fieldAbove = 60.0f;    ///< height over the far end of the field
    float lowAbove = 8.0f;       ///< height over the field before the forest
    float lowBefore = 50.0f;     ///< ... that far before the edge
    float forestAbove = 2.0f;    ///< height in the forest
    float eyeAbove = 1.7f;       ///< height at the end
    float trunkClearance = 0.7f; ///< from every stem of the trees near the way (the Gaussian renderer's)
    float leafClearance = 0.35f; ///< from every leaf (beyond the near plane; leaves hang everywhere in the forest)
    float soilClearance = 0.5f;
    float cruise = 25.0f;  ///< the way left from which on the camera keeps its speed, at least the forest (edge + 5 m)
    float zoom = 1.3f;     ///< before, its speed grows with the way left to this power
    float seconds = 24.0f; ///< how long the flight (u from 0 to 1) is played
    float maxTurn = 60.0f; ///< degrees per second the way may turn at most; the camera slows down for bends
    float season = 2.0f;   ///< of the trees' leaves
    uint32_t variants = 8; ///< of the trees' seeds (TreeGaussianForestConfig::variants)
};

/// A place in dense forest with open field behind it: the end (x, z in units), the heading (radians from -z towards
/// +x) the camera looks along there, and how far behind the end (along the heading) the forest's edge is, in metres.
struct ForestApproach {
    glm::vec2 end{0.0f};
    float heading = 0.0f;
    float edgeMeters = 0.0f;
};

/// The places of the scene where a flight can cross open field into dense forest, best first (few conifers, whose
/// branches reach the ground; the forest's edge near; the ground level, gentle under the low part of the flight):
/// forest all around the end, the edge at most maxEdge behind it and fieldLength of open field behind the edge, all
/// inside the pot.
std::vector<ForestApproach> findForestApproach(const BonsaiScene& scene, ForestApproachConfig config = {});

/**
 * @brief A flight from `start` (looking at `focus`) over a field into a forest (findForestApproach()).
 *
 * The camera flies along a centripetal Catmull-Rom spline: from the start to fieldAbove over the far end of the
 * field, down to lowAbove over it lowBefore the edge, from 20 m before the edge at 3.5 m into the forest at
 * forestAbove, and to eyeAbove at the end. The way between the trees is found by A* on a 0.25 m grid of the room
 * their stems and leaves leave 0.8 to 4 m above the soil (the camera's height band), keeping trunkClearance from
 * stems and leafClearance from leaves and preferring room; it is pulled taut (straight between corners that keep an
 * extra margin of room) and relaxed as an elastic band into round curves, the margin growing until the spline keeps
 * its clearances in 3D. The first of the best 12 approaches whose way does is taken.
 *
 * Far away the camera's speed grows with the way left (to the power `zoom`: a little faster than each halving
 * taking the same time, which leaves time for the forest); from `cruise` before the end (at least from 5 m before
 * the edge) it keeps the speed it has there, slows down where the way would turn faster than maxTurn, and brakes
 * before the end, which it reaches at u = 0.85. It looks at the place, then 10 to 34 m ahead along its way, and at
 * the end along its way's last 10 m, panning slowly by 35 degrees until u = 1.
 */
class ForestFlight {
public:
    ForestFlight(const BonsaiScene& scene, glm::vec3 start, glm::vec3 focus,
                 const std::vector<ForestApproach>& approaches, ForestApproachConfig config = {});

    FlyInPose at(float u) const;
    bool isValid() const { return valid; }
    const ForestApproach& getApproach() const { return approach; }
    /// The spline's control points, from the start to the end.
    const std::vector<glm::vec3>& getWaypoints() const { return waypoints; }
    /// How far `p` is from the nearest stem or leaf (up to 15 m above the soil) of the trees near the way
    /// (generated as the Gaussian renderer grows them, treeInstanceFor() with `variants`), in metres, the stems
    /// widened by trunkClearance - leafClearance; large far from them.
    float clearanceAt(glm::vec3 p) const;

private:
    bool build(const ForestApproach& candidate);
    bool buildWay(const ForestApproach& candidate, const std::vector<glm::vec2>& way, float margin);
    void gatherTrees(const ForestApproach& candidate);
    /// The room in metres at `p` on the planning grid (beyond it: much).
    float roomAt(glm::vec2 p) const;
    glm::vec3 positionAt(float remaining) const;

    const BonsaiScene& scene;
    glm::vec3 start;
    glm::vec3 focus;
    ForestApproachConfig config;
    float unitsPerMeter;
    bool valid = false;
    ForestApproach approach;
    std::vector<glm::vec3> waypoints;
    std::vector<glm::vec4> points;              ///< stems and leaves near the way: xyz, radius (units)
    std::vector<std::vector<uint32_t>> buckets; ///< `points` by 2 m cell of the grid's frame
    // The planning grid in the frame of the heading: origin, axes, cells (0.25 m), room per cell in metres.
    glm::vec2 gridOrigin{0.0f}, gridForward{0.0f}, gridSide{0.0f};
    float gridFrom = 0.0f, gridWidth = 0.0f, gridCell = 0.25f;
    int gridNx = 0, gridNy = 0;
    std::vector<float> room;
    std::vector<glm::vec3> samples; ///< along the spline, by arc length
    std::vector<float> lengths;     ///< arc length up to each sample
    std::vector<float> remainingAt; ///< the way left at u = k / (timeSteps - 1) * 0.85
    float total = 0.0f;
    float lowRemaining = 0.0f;
    float endHeading = 0.0f; ///< along the way's last 10 m
};

} // namespace klartraum

#endif // KLARTRAUM_TREE_BONSAI_FLIGHT_HPP
