// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/tree/bonsai_flight.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>

#include <glm/gtc/quaternion.hpp>

#include "klartraum/tree/tree_placement.hpp"

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

namespace {

glm::vec2 forwardOf(float heading) { return glm::vec2(std::sin(heading), -std::cos(heading)); }

float forestAt(const BonsaiScene& scene, glm::vec2 p) {
    return tree::treeLandCover(scene.terrain, p.x, p.y).vegetation.x;
}

bool inPot(const BonsaiScene& scene, glm::vec2 p) { return tree::treeTerrainRadius(scene.terrain, p.x, p.y) < 0.9f; }

// Centripetal Catmull-Rom between p1 and p2 at t in [0, 1].
glm::vec3 catmullRom(glm::vec3 p0, glm::vec3 p1, glm::vec3 p2, glm::vec3 p3, float t) {
    auto knot = [](float k, glm::vec3 a, glm::vec3 b) { return k + std::sqrt(std::max(glm::length(b - a), 1e-9f)); };
    const float t0 = 0.0f, t1 = knot(t0, p0, p1), t2 = knot(t1, p1, p2), t3 = knot(t2, p2, p3);
    const float x = glm::mix(t1, t2, t);
    const glm::vec3 a1 = ((t1 - x) * p0 + (x - t0) * p1) / (t1 - t0);
    const glm::vec3 a2 = ((t2 - x) * p1 + (x - t1) * p2) / (t2 - t1);
    const glm::vec3 a3 = ((t3 - x) * p2 + (x - t2) * p3) / (t3 - t2);
    const glm::vec3 b1 = ((t2 - x) * a1 + (x - t0) * a2) / (t2 - t0);
    const glm::vec3 b2 = ((t3 - x) * a2 + (x - t1) * a3) / (t3 - t1);
    return ((t2 - x) * b1 + (x - t1) * b2) / (t2 - t1);
}

} // namespace

std::vector<ForestApproach> findForestApproach(const BonsaiScene& scene, ForestApproachConfig config) {
    const float m = 1.0f / scene.config.scale.metersPerUnit();
    const glm::vec2 half(scene.terrain.outline.x, scene.terrain.outline.y);
    const float step = 40.0f * m;
    struct Scored {
        ForestApproach approach;
        float score;
    };
    std::vector<Scored> found;
    for (float x = -half.x; x <= half.x; x += step) {
        for (float z = -half.y; z <= half.y; z += step) {
            const glm::vec2 end(x, z);
            if (!inPot(scene, end) || forestAt(scene, end) < config.minForest) {
                continue;
            }
            bool all = true;
            for (int ring = 1; ring <= 3 && all; ++ring) {
                for (int k = 0; k < 8 && all; ++k) {
                    const float a = 2.0f * 3.14159265f * (float(k) + 0.5f * float(ring)) / 8.0f;
                    const glm::vec2 p =
                        end + config.forestRadius * m * float(ring) / 3.0f * glm::vec2(std::cos(a), std::sin(a));
                    all = forestAt(scene, p) >= config.minForest;
                }
            }
            if (!all) {
                continue;
            }
            for (int h = 0; h < 32; ++h) {
                const float heading = 2.0f * 3.14159265f * float(h) / 32.0f;
                const glm::vec2 back = -forwardOf(heading);
                float edge = -1.0f;
                for (float d = config.forestRadius; d <= config.maxEdge; d += 5.0f) {
                    if (forestAt(scene, end + d * m * back) < 0.5f) {
                        edge = d;
                        break;
                    }
                }
                if (edge < 0.0f) {
                    continue;
                }
                bool open = true;
                for (float d = edge + 5.0f; d <= edge + config.fieldLength + config.lowBefore && open; d += 10.0f) {
                    const glm::vec2 p = end + d * m * back;
                    const glm::vec4 cover = tree::treeLandCover(scene.terrain, p.x, p.y).vegetation;
                    open = inPot(scene, p) && cover.x < 0.2f;
                }
                if (!open) {
                    continue;
                }
                // The steepest ground under the low part of the flight, over 10 m.
                float steepest = 0.0f;
                for (float d = 0.0f; d < edge + config.lowBefore; d += 5.0f) {
                    const glm::vec2 p = end + d * m * back, q = end + (d + 10.0f) * m * back;
                    steepest = std::max(steepest, std::abs(scene.soilHeight(q.x, q.y) - scene.soilHeight(p.x, p.y)) /
                                                      (10.0f * m));
                }
                const glm::vec2 far = end + (edge + config.fieldLength) * m * back;
                const float rise = std::abs(scene.soilHeight(far.x, far.y) - scene.soilHeight(end.x, end.y)) / m;
                // Conifers' branches reach the ground; the fewer there, the more room for the camera.
                const float conifers =
                    tree::treeConiferShare(scene.placement, tree::treeLandSite(scene.terrain, end.x, end.y).y);
                found.push_back({{end, heading, edge}, edge + 0.5f * rise + 100.0f * steepest + 300.0f * conifers});
            }
        }
    }
    std::sort(found.begin(), found.end(), [](const Scored& a, const Scored& b) { return a.score < b.score; });
    std::vector<ForestApproach> approaches;
    for (const Scored& s : found) {
        approaches.push_back(s.approach);
    }
    return approaches;
}

ForestFlight::ForestFlight(const BonsaiScene& scene, glm::vec3 start, glm::vec3 focus,
                           const std::vector<ForestApproach>& approaches, ForestApproachConfig config)
    : scene(scene),
      start(start),
      focus(focus),
      config(config),
      unitsPerMeter(1.0f / scene.config.scale.metersPerUnit()) {
    // The best few; each takes a moment to plan.
    for (size_t k = 0; k < std::min<size_t>(approaches.size(), 12); ++k) {
        const ForestApproach& candidate = approaches[k];
        if (build(candidate)) {
            approach = candidate;
            valid = true;
            return;
        }
    }
}

bool ForestFlight::build(const ForestApproach& candidate) {
    const float m = unitsPerMeter;
    const glm::vec2 fwd = forwardOf(candidate.heading);
    const glm::vec2 side(-fwd.y, fwd.x);
    const glm::vec2 end = candidate.end;
    const float edge = candidate.edgeMeters;
    auto at = [&](float along, float across) { return end + m * (along * fwd + across * side); };

    // The room between the trees near the way, on a grid in the frame of the heading: from 20 m before the edge
    // to the end.
    gatherTrees(candidate);
    const float cell = gridCell, from = gridFrom, width = gridWidth;
    const int nx = gridNx, ny = gridNy;
    auto world = [&](int i, int j) { return at(from + float(i) * cell, -width + float(j) * cell); };
    const float needed = config.leafClearance + 0.15f;
    auto free = [&](int i, int j) { return room[size_t(i) * ny + j] >= needed; };
    const int si = 0, sj = ny / 2;
    // The end: the cell nearest to the place (within 6 m) with half a metre more room than the clearance.
    int gi = -1, gj = -1;
    float best = 1e9f;
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            const float d = glm::length(glm::vec2(from + float(i) * cell, -width + float(j) * cell));
            if (room[size_t(i) * ny + j] >= config.leafClearance + 0.5f && d < best && d < 6.0f) {
                best = d;
                gi = i;
                gj = j;
            }
        }
    }
    if (gi < 0 || !free(si, sj)) {
        return false;
    }
    std::vector<float> cost(size_t(nx) * ny, std::numeric_limits<float>::infinity());
    std::vector<int> parent(size_t(nx) * ny, -1);
    using Node = std::pair<float, int>;
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> open;
    cost[size_t(si) * ny + sj] = 0.0f;
    open.push({0.0f, si * ny + sj});
    const int goal = gi * ny + gj;
    while (!open.empty()) {
        const auto [f, node] = open.top();
        open.pop();
        if (node == goal) {
            break;
        }
        const int i = node / ny, j = node % ny;
        for (int di = -1; di <= 1; ++di) {
            for (int dj = -1; dj <= 1; ++dj) {
                const int a = i + di, b = j + dj;
                if ((di == 0 && dj == 0) || a < 0 || b < 0 || a >= nx || b >= ny || !free(a, b)) {
                    continue;
                }
                // Longer ways cost, and so does little room.
                const float step = cell * std::sqrt(float(di * di + dj * dj));
                const float c = cost[size_t(node)] + step * (1.0f + 1.5f / std::max(room[size_t(a) * ny + b], 0.2f));
                if (c < cost[size_t(a) * ny + b]) {
                    cost[size_t(a) * ny + b] = c;
                    parent[size_t(a) * ny + b] = node;
                    const float h = cell * glm::length(glm::vec2(float(a - gi), float(b - gj)));
                    open.push({c + h, a * ny + b});
                }
            }
        }
    }
    if (parent[size_t(goal)] < 0) {
        return false;
    }
    std::vector<glm::vec2> way;
    for (int node = goal; node >= 0; node = parent[size_t(node)]) {
        way.push_back(world(node / ny, node % ny));
    }
    std::reverse(way.begin(), way.end());
    for (float margin : {0.4f, 0.8f, 1.2f, 1.6f}) {
        if (buildWay(candidate, way, margin)) {
            return true;
        }
    }
    return false;
}

bool ForestFlight::buildWay(const ForestApproach& candidate, const std::vector<glm::vec2>& way, float margin) {
    const float m = unitsPerMeter;
    const glm::vec2 fwd = forwardOf(candidate.heading);
    const glm::vec2 side(-fwd.y, fwd.x);
    const glm::vec2 end = candidate.end;
    const float edge = candidate.edgeMeters;
    auto at = [&](float along, float across) { return end + m * (along * fwd + across * side); };
    auto above = [&](glm::vec2 p, float meters) {
        return glm::vec3(p.x, scene.soilHeight(p.x, p.y) + meters * m, p.y);
    };
    // Whether the straight line from a to b keeps the clearance and `margin` on the grid.
    auto clear = [&](glm::vec2 a, glm::vec2 b) {
        const int steps = std::max(1, int(glm::length(b - a) / (0.25f * m)));
        for (int k = 0; k <= steps; ++k) {
            if (roomAt(glm::mix(a, b, float(k) / float(steps))) < config.leafClearance + margin) {
                return false;
            }
        }
        return true;
    };
    // The way pulled taut: from each corner straight to the farthest point of the way still in clear sight.
    std::vector<glm::vec2> corners = {way.front()};
    size_t k = 0;
    while (k + 1 < way.size()) {
        size_t next = k + 1;
        for (size_t j = way.size() - 1; j > k + 1; --j) {
            if (clear(way[k], way[j])) {
                next = j;
                break;
            }
        }
        corners.push_back(way[next]);
        k = next;
    }
    // An elastic band along it, a point per metre: each pulled towards its neighbours' middle and pushed out of
    // the discs grown by the margin, the ends held, until it settles into round curves.
    std::vector<glm::vec2> band = {corners.front()};
    for (size_t c = 1; c < corners.size(); ++c) {
        const int pieces = std::max(1, int(std::ceil(glm::length(corners[c] - corners[c - 1]) / m)));
        for (int q = 1; q <= pieces; ++q) {
            band.push_back(glm::mix(corners[c - 1], corners[c], float(q) / float(pieces)));
        }
    }
    const float needed = config.leafClearance + margin;
    const float e = gridCell * m;
    for (int iteration = 0; iteration < 400; ++iteration) {
        std::vector<glm::vec2> next = band;
        for (size_t q = 1; q + 1 < band.size(); ++q) {
            glm::vec2 p = band[q] + 0.4f * (0.5f * (band[q - 1] + band[q + 1]) - band[q]);
            // Out of tight places, up the room's gradient.
            const float r = roomAt(p);
            if (r < needed) {
                const glm::vec2 g(roomAt(p + glm::vec2(e, 0.0f)) - roomAt(p - glm::vec2(e, 0.0f)),
                                  roomAt(p + glm::vec2(0.0f, e)) - roomAt(p - glm::vec2(0.0f, e)));
                if (glm::length(g) > 1e-9f) {
                    p += 0.5f * (needed - r) * m * glm::normalize(g);
                }
            }
            next[q] = p;
        }
        band = next;
    }
    // The ground it flies over: the highest soil within 6 m, averaged over 12 m, so that the camera does not ride
    // every bump; over the last 8 m it comes down to the soil itself.
    const int n1 = int(band.size());
    std::vector<float> soil(band.size()), highest(band.size()), ground(band.size());
    for (int q = 0; q < n1; ++q) {
        soil[q] = scene.soilHeight(band[q].x, band[q].y);
    }
    for (int q = 0; q < n1; ++q) {
        highest[q] = soil[q];
        for (int j = std::max(q - 6, 0); j <= std::min(q + 6, n1 - 1); ++j) {
            highest[q] = std::max(highest[q], soil[j]);
        }
    }
    for (int q = 0; q < n1; ++q) {
        float sum = 0.0f;
        int count = 0;
        for (int j = std::max(q - 12, 0); j <= std::min(q + 12, n1 - 1); ++j) {
            sum += highest[j];
            ++count;
        }
        ground[q] = glm::mix(soil[q], sum / float(count), smootherstep(float(n1 - 1 - q) / 8.0f));
    }
    // The camera looks along the way's last 10 m at the end.
    const glm::vec2 arrival = band.back() - band[size_t(std::max(n1 - 11, 0))];
    endHeading = glm::length(arrival) > 1e-9f ? std::atan2(arrival.x, -arrival.y) : candidate.heading;
    // Every 3 m of it for the spline.
    std::vector<int> taut;
    for (int q = 0; q < n1; q += 3) {
        taut.push_back(q);
    }
    taut.back() = n1 - 1;

    // The control points: the start, high over the far field, low over the field, then the taut way.
    // Over the field the camera keeps its height over the highest soil within 15 m along the way.
    auto overField = [&](float along, float meters) {
        float highest = -1e30f;
        for (float d = -15.0f; d <= 15.0f; d += 2.5f) {
            const glm::vec2 p = at(along + d, 0.0f);
            highest = std::max(highest, scene.soilHeight(p.x, p.y));
        }
        const glm::vec2 p = at(along, 0.0f);
        return glm::vec3(p.x, highest + meters * m, p.y);
    };
    waypoints = {start, overField(-(edge + config.fieldLength), config.fieldAbove),
                 overField(-(edge + config.lowBefore), config.lowAbove)};
    // Along the way: down from 3.5 m over the field to forestAbove at the edge, then to eyeAbove at the end.
    for (size_t q = 0; q < taut.size(); ++q) {
        const glm::vec2 p = band[size_t(taut[q])];
        const float along = glm::dot((p - end) / m, fwd);
        const float meters = along < -edge ? glm::mix(config.forestAbove, 3.5f, smootherstep((-along - edge) / 20.0f))
                                           : glm::mix(config.forestAbove, config.eyeAbove,
                                                      std::pow(std::clamp((along + edge) / edge, 0.0f, 1.0f), 2.0f));
        waypoints.push_back(glm::vec3(p.x, ground[size_t(taut[q])] + meters * m, p.y));
    }

    // The spline by arc length.
    samples.clear();
    lengths.clear();
    const size_t n = waypoints.size();
    for (size_t k = 0; k + 1 < n; ++k) {
        const glm::vec3 p0 = k == 0 ? 2.0f * waypoints[0] - waypoints[1] : waypoints[k - 1];
        const glm::vec3 p3 = k + 2 < n ? waypoints[k + 2] : 2.0f * waypoints[n - 1] - waypoints[n - 2];
        for (int s = 0; s < 400; ++s) {
            samples.push_back(catmullRom(p0, waypoints[k], waypoints[k + 1], p3, float(s) / 400.0f));
        }
    }
    samples.push_back(waypoints.back());
    lengths.assign(samples.size(), 0.0f);
    for (size_t k = 1; k < samples.size(); ++k) {
        lengths[k] = lengths[k - 1] + glm::length(samples[k] - samples[k - 1]);
    }
    total = lengths.back();

    // The clearances along the way.
    for (const glm::vec3& p : samples) {
        if (p.y - scene.soilHeight(p.x, p.z) < config.soilClearance * m) {
            return false;
        }
        if (clearanceAt(p) < config.leafClearance) {
            return false;
        }
    }

    // The timing: speed k * cruise * (max(left, cruise) / cruise)^zoom, cruising through all of the forest,
    // braking within 5 m of the end; k such that the end comes at u = 0.85.
    lowRemaining = std::max(config.cruise, candidate.edgeMeters + 5.0f) * m;
    const float brake = 5.0f * m;
    const int steps = 8192;
    // How fast the way turns (radians per unit of length) around each sample, over 1 m either side.
    std::vector<float> bend(samples.size(), 0.0f);
    {
        size_t back = 0, ahead = 0;
        for (size_t k = 0; k < samples.size(); ++k) {
            while (lengths[k] - lengths[back] > 1.0f * m) {
                ++back;
            }
            while (ahead + 1 < samples.size() && lengths[ahead] - lengths[k] < 1.0f * m) {
                ++ahead;
            }
            const glm::vec3 a = samples[k] - samples[back], b = samples[ahead] - samples[k];
            const float span = lengths[ahead] - lengths[back];
            if (glm::length(a) > 1e-9f && glm::length(b) > 1e-9f && span > 1e-9f) {
                bend[k] = std::acos(std::clamp(glm::dot(glm::normalize(a), glm::normalize(b)), -1.0f, 1.0f)) / span;
            }
        }
    }
    // The largest speed (way per unit of u) at which the way turns at most maxTurn degrees per second of a flight
    // of `seconds`, changing along the way by at most a factor e per 8 m, so that the camera slows down gently
    // before a bend and speeds up gently after it.
    const float turnPerU = glm::radians(config.maxTurn) * config.seconds;
    std::vector<float> limit(samples.size());
    for (size_t k = 0; k < samples.size(); ++k) {
        limit[k] = bend[k] > 1e-9f ? turnPerU / bend[k] : 1e30f;
    }
    const float ease = 8.0f * m;
    for (size_t k = samples.size() - 1; k > 0; --k) {
        limit[k - 1] = std::min(limit[k - 1], limit[k] * std::exp((lengths[k] - lengths[k - 1]) / ease));
    }
    for (size_t k = 1; k < samples.size(); ++k) {
        limit[k] = std::min(limit[k], limit[k - 1] * std::exp((lengths[k] - lengths[k - 1]) / ease));
    }
    auto speedLimit = [&](float left) {
        const float along = total - left;
        const size_t k = size_t(std::lower_bound(lengths.begin(), lengths.end(), along) - lengths.begin());
        return limit[std::min(k, limit.size() - 1)];
    };
    auto simulate = [&](float k, std::vector<float>* out) {
        float left = total;
        const float dt = 0.85f / float(steps - 1);
        for (int i = 0; i < steps; ++i) {
            if (out) {
                out->push_back(left);
            }
            const float t = float(i) * dt;
            const float ramp = smootherstep(t / 0.06f);
            const float speed =
                k * (left > brake ? lowRemaining * std::pow(std::max(left, lowRemaining) / lowRemaining, config.zoom)
                                  : lowRemaining * left / brake);
            left = std::max(left - std::max(std::min(speed, speedLimit(left)) * ramp, 0.0f) * dt, 0.0f);
        }
        return left;
    };
    float lo = 0.1f, hi = 1000.0f;
    for (int it = 0; it < 60; ++it) {
        const float mid = std::sqrt(lo * hi);
        if (simulate(mid, nullptr) > 0.02f * m) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    remainingAt.clear();
    simulate(hi, &remainingAt);
    return true;
}

void ForestFlight::gatherTrees(const ForestApproach& candidate) {
    const float m = unitsPerMeter;
    gridOrigin = candidate.end;
    gridForward = forwardOf(candidate.heading);
    gridSide = glm::vec2(-gridForward.y, gridForward.x);
    gridFrom = -(candidate.edgeMeters + 20.0f);
    gridWidth = 12.0f;
    gridNx = int((2.0f - gridFrom) / gridCell) + 1;
    gridNy = int(2.0f * gridWidth / gridCell) + 1;

    // Each type's trees of the library's seeds at scale 1, as points along the stems and at the leaves: x, y, z
    // and radius in the type's metres.
    static std::vector<std::vector<glm::vec4>> grown;
    const uint32_t variants = std::max(config.variants, 1u);
    if (grown.size() != scene.types.size() * variants) {
        grown.assign(scene.types.size() * variants, {});
        TreeFrame frame = defaultTreeFrame();
        frame.wind.w = 0.0f;
        frame.timeSeason.y = config.season;
        for (uint32_t k = 0; k < grown.size(); ++k) {
            TreeInstance inst{};
            inst.posScale = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
            inst.rot = treeUpRotation(glm::vec3(0.0f, 1.0f, 0.0f));
            inst.ids = glm::uvec4(k / variants, 1u + k % variants, 0u, 0u);
            const TreeRecords records = generateTreesReference(scene.types, {inst}, frame);
            for (const SegmentRecord& seg : records.segments) {
                const glm::vec3 a(seg.fromPosRadius), b(seg.toPosRadius);
                const float r = std::max(seg.fromPosRadius.w, seg.toPosRadius.w);
                const int n = std::max(1, int(std::ceil(glm::length(b - a) / std::max(r, 0.05f))));
                for (int q = 0; q <= n; ++q) {
                    grown[k].push_back(glm::vec4(glm::mix(a, b, float(q) / float(n)), r));
                }
            }
            for (const LeafRecord& leaf : records.leaves) {
                const float length = leaf.posScale.w;
                // Leaves carry their radius negated: the camera may come closer to them than to the stems.
                grown[k].push_back(
                    glm::vec4(glm::vec3(leaf.posScale) + 0.5f * length * tree::qforward(leaf.rot), -0.5f * length));
            }
        }
    }

    // The trees near the grid, placed as treeInstanceFor() places them.
    const glm::vec2 middle = candidate.end + 0.5f * gridFrom * m * gridForward;
    const std::vector<TreeSite> sites = nearestTreeSites(
        scene.terrain, scene.placement, glm::vec3(middle.x, scene.soilHeight(middle.x, middle.y), middle.y), 3000);
    const float bucketFrom = gridFrom - 10.0f, bucketTo = 12.0f, bucketSide = gridWidth + 10.0f;
    const int bx = int((bucketTo - bucketFrom) / 2.0f) + 1, by = int(2.0f * bucketSide / 2.0f) + 1;
    buckets.assign(size_t(bx) * by, {});
    points.clear();
    room.assign(size_t(gridNx) * gridNy, 1e9f);
    for (const TreeSite& site : sites) {
        const glm::vec2 foot(site.posHeight.x, site.posHeight.z);
        const glm::vec2 local = (foot - gridOrigin) / m;
        if (glm::dot(local, gridForward) < bucketFrom || glm::dot(local, gridForward) > bucketTo ||
            std::abs(glm::dot(local, gridSide)) > bucketSide) {
            continue;
        }
        const TreeInstance inst = treeInstanceFor(site, scene.types, scene.config.seed, variants);
        const std::vector<glm::vec4>& tree = grown[inst.ids.x * variants + (inst.ids.y - 1u)];
        const glm::mat3 turn = glm::mat3_cast(glm::angleAxis(site.kind.y, glm::vec3(0.0f, 1.0f, 0.0f)));
        const float scale = inst.posScale.w;
        for (const glm::vec4& q : tree) {
            const glm::vec3 p = glm::vec3(site.posHeight) + turn * (scale * glm::vec3(q));
            // Stems are widened by what the camera keeps from them beyond the leaves' clearance.
            const float radius = q.w < 0.0f
                                     ? -scale * q.w
                                     : scale * q.w + std::max(config.trunkClearance - config.leafClearance, 0.0f) * m;
            // The camera stays low in the forest; crowns high above it do not matter.
            const float above = (p.y - scene.soilHeight(p.x, p.z)) / m;
            if (above > 15.0f) {
                continue;
            }
            const glm::vec2 l = (glm::vec2(p.x, p.z) - gridOrigin) / m;
            const float along = glm::dot(l, gridForward), across = glm::dot(l, gridSide);
            const int i = int((along - bucketFrom) / 2.0f), j = int((across + bucketSide) / 2.0f);
            if (i < 0 || j < 0 || i >= bx || j >= by) {
                continue;
            }
            buckets[size_t(i) * by + j].push_back(uint32_t(points.size()));
            points.push_back(glm::vec4(p, radius));
            // What stands where the camera flies marks the grid: the cells it covers have no room.
            if (above < 0.8f || above > 4.0f) {
                continue;
            }
            const float r = radius / m;
            const int i0 = int(std::floor((along - r - gridFrom) / gridCell)),
                      i1 = int(std::ceil((along + r - gridFrom) / gridCell));
            const int j0 = int(std::floor((across - r + gridWidth) / gridCell)),
                      j1 = int(std::ceil((across + r + gridWidth) / gridCell));
            for (int a = std::max(i0, 0); a <= std::min(i1, gridNx - 1); ++a) {
                for (int b = std::max(j0, 0); b <= std::min(j1, gridNy - 1); ++b) {
                    const glm::vec2 c(gridFrom + float(a) * gridCell - along,
                                      -gridWidth + float(b) * gridCell - across);
                    if (glm::length(c) <= r + 0.5f * gridCell) {
                        room[size_t(a) * gridNy + b] = 0.0f;
                    }
                }
            }
        }
    }
    // The room: the distance to the nearest marked cell (chamfer distance, two passes).
    const float straight = gridCell, diagonal = gridCell * std::sqrt(2.0f);
    auto relax = [&](int i, int j, int a, int b, float w) {
        if (a >= 0 && b >= 0 && a < gridNx && b < gridNy) {
            float& r = room[size_t(i) * gridNy + j];
            r = std::min(r, room[size_t(a) * gridNy + b] + w);
        }
    };
    for (int i = 0; i < gridNx; ++i) {
        for (int j = 0; j < gridNy; ++j) {
            relax(i, j, i - 1, j, straight);
            relax(i, j, i, j - 1, straight);
            relax(i, j, i - 1, j - 1, diagonal);
            relax(i, j, i - 1, j + 1, diagonal);
        }
    }
    for (int i = gridNx - 1; i >= 0; --i) {
        for (int j = gridNy - 1; j >= 0; --j) {
            relax(i, j, i + 1, j, straight);
            relax(i, j, i, j + 1, straight);
            relax(i, j, i + 1, j + 1, diagonal);
            relax(i, j, i + 1, j - 1, diagonal);
        }
    }
}

float ForestFlight::roomAt(glm::vec2 p) const {
    const glm::vec2 l = (p - gridOrigin) / unitsPerMeter;
    const int i = int(std::lround((glm::dot(l, gridForward) - gridFrom) / gridCell));
    const int j = int(std::lround((glm::dot(l, gridSide) + gridWidth) / gridCell));
    if (i < 0 || j < 0 || i >= gridNx || j >= gridNy) {
        return 1e9f;
    }
    return room[size_t(i) * gridNy + j];
}

float ForestFlight::clearanceAt(glm::vec3 p) const {
    const float m = unitsPerMeter;
    const float bucketFrom = gridFrom - 10.0f, bucketTo = 12.0f, bucketSide = gridWidth + 10.0f;
    const int bx = int((bucketTo - bucketFrom) / 2.0f) + 1, by = int(2.0f * bucketSide / 2.0f) + 1;
    const glm::vec2 l = (glm::vec2(p.x, p.z) - gridOrigin) / m;
    const int ci = int((glm::dot(l, gridForward) - bucketFrom) / 2.0f),
              cj = int((glm::dot(l, gridSide) + bucketSide) / 2.0f);
    float clearance = 1e9f;
    for (int i = ci - 2; i <= ci + 2; ++i) {
        for (int j = cj - 2; j <= cj + 2; ++j) {
            if (i < 0 || j < 0 || i >= bx || j >= by) {
                continue;
            }
            for (uint32_t k : buckets[size_t(i) * by + j]) {
                clearance = std::min(clearance, (glm::length(glm::vec3(points[k]) - p) - points[k].w) / m);
            }
        }
    }
    return clearance;
}

glm::vec3 ForestFlight::positionAt(float remaining) const {
    const float along = std::clamp(total - remaining, 0.0f, total);
    const auto it = std::lower_bound(lengths.begin(), lengths.end(), along);
    const size_t k = size_t(std::clamp<std::ptrdiff_t>(it - lengths.begin(), 1, std::ptrdiff_t(lengths.size()) - 1));
    const float span = std::max(lengths[k] - lengths[k - 1], 1e-12f);
    return glm::mix(samples[k - 1], samples[k], std::clamp((along - lengths[k - 1]) / span, 0.0f, 1.0f));
}

FlyInPose ForestFlight::at(float u) const {
    const float m = unitsPerMeter;
    const float x = std::clamp(u / 0.85f, 0.0f, 1.0f) * float(remainingAt.size() - 1);
    const size_t i = std::min(size_t(x), remainingAt.size() - 2);
    const float left = glm::mix(remainingAt[i], remainingAt[i + 1], x - float(i));
    const glm::vec3 eye = positionAt(left);

    // How far the flight has come in halvings of the way, up to the low point.
    const float halvings = std::log(std::max(total, 1e-9f) / std::max(lowRemaining, 1e-9f));
    const float p =
        left > lowRemaining ? std::clamp(1.0f - std::log(left / lowRemaining) / halvings, 0.0f, 1.0f) : 1.0f;
    const glm::vec3 place = waypoints.back();
    const glm::vec3 toPlace = glm::normalize(glm::mix(focus, place, smootherstep(p / 0.3f)) - eye);
    // Along the way, a little down; at the end along the heading, panning.
    // Towards the mean of the way 10 to 34 m ahead, which smooths the turns between the trunks.
    glm::vec3 target(0.0f);
    // Beyond the end the way goes on straight along the end's heading.
    const glm::vec2 onward = forwardOf(endHeading);
    for (float d : {10.0f, 18.0f, 26.0f, 34.0f}) {
        const float beyond = std::max(d * m - left, 0.0f);
        target += 0.25f * (positionAt(std::max(left - d * m, 0.0f)) + beyond * glm::vec3(onward.x, 0.0f, onward.y));
    }
    const glm::vec3 ahead = target - eye;
    const float pan = smootherstep((u - 0.85f) / 0.15f);
    const glm::vec2 f = forwardOf(endHeading + glm::radians(35.0f) * pan);
    const glm::vec3 heading = glm::normalize(glm::vec3(f.x, -0.08f, f.y));
    glm::vec3 along = glm::length(ahead) > 1e-9f
                          ? glm::normalize(ahead + glm::vec3(0.0f, -0.08f, 0.0f) * glm::length(ahead))
                          : heading;
    along = slerpDirection(along, heading, 1.0f - smootherstep(left / (15.0f * m)));
    const glm::vec3 look = slerpDirection(toPlace, along, smootherstep((p - 0.3f) / 0.7f));
    const float clearance = eye.y - scene.soilHeight(eye.x, eye.z);
    return {eye, eye + look, std::clamp(std::min(0.5f * clearance, 0.3f * m), 1e-5f, 0.05f)};
}

} // namespace klartraum
