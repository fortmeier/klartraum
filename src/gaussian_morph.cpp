// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/gaussian_morph.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "klartraum/gaussian_transform.hpp"

namespace klartraum {

void placeFacingCamera(std::vector<Gaussian3D>& gaussians, const glm::mat4& view, const glm::vec3& centre) {
    // The columns of the inverse view rotation are the camera's right, up and
    // backward directions in world space; they take the text plane's x, y and
    // its normal +z.
    const glm::mat3 cameraToWorld = glm::inverse(glm::mat3(view));
    transformGaussians(gaussians, glm::normalize(glm::quat_cast(cameraToWorld)), 1.0f, centre);
}

namespace {

// Indices of `gaussians` in reading order for the camera of `view`: into
// `bands` rows of equal size from top to bottom, each row from left to right.
// View-space x and y are used directly (an orthographic view), so splats
// behind the camera still get a place.
std::vector<uint32_t> readingOrder(const std::vector<Gaussian3D>& gaussians, const glm::mat4& view, size_t bands) {
    const size_t n = gaussians.size();
    std::vector<glm::vec2> screen(n);
    for (size_t i = 0; i < n; ++i) {
        const auto& p = gaussians[i].position;
        screen[i] = glm::vec2(view * glm::vec4(p[0], p[1], p[2], 1.0f));
    }
    std::vector<uint32_t> order(n);
    for (uint32_t i = 0; i < n; ++i) {
        order[i] = i;
    }
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        return screen[a].y != screen[b].y ? screen[a].y > screen[b].y : screen[a].x < screen[b].x;
    });
    for (size_t band = 0; band < bands; ++band) {
        const auto begin = order.begin() + std::ptrdiff_t(band * n / bands);
        const auto end = order.begin() + std::ptrdiff_t((band + 1) * n / bands);
        std::sort(begin, end, [&](uint32_t a, uint32_t b) {
            return screen[a].x != screen[b].x ? screen[a].x < screen[b].x : screen[a].y > screen[b].y;
        });
    }
    return order;
}

} // namespace

std::vector<uint32_t> matchByScreenPosition(const std::vector<Gaussian3D>& text, const std::vector<Gaussian3D>& scene,
                                            const glm::mat4& view) {
    if (text.size() != scene.size()) {
        throw std::invalid_argument("matchByScreenPosition: both sets need the same number of Gaussians");
    }
    // About as many bands as Gaussians per band.
    const size_t bands = std::max<size_t>(1, size_t(std::lround(std::sqrt(double(scene.size())))));
    const std::vector<uint32_t> textOrder = readingOrder(text, view, bands);
    const std::vector<uint32_t> sceneOrder = readingOrder(scene, view, bands);
    std::vector<uint32_t> match(scene.size());
    for (size_t rank = 0; rank < scene.size(); ++rank) {
        match[sceneOrder[rank]] = textOrder[rank];
    }
    return match;
}

std::vector<float> MorphParams::values() const {
    return {progress,      float(int(variant)),
            swirlTurns,    scatterRadius,
            waveSpread,    centre.x,
            centre.y,      centre.z,
            axis.x,        axis.y,
            axis.z,        float(seed),
            explodeRadius, explodeShare,
            direction.x,   direction.y,
            direction.z};
}

uint32_t MorphTimeline::frameCount() const {
    return uint32_t(std::lround(double(titleSeconds + morphSeconds + spinSeconds) * fps));
}

uint32_t MorphTimeline::titleFrames() const { return uint32_t(std::ceil(double(titleSeconds) * fps - 1e-6)); }

float MorphTimeline::progressAt(uint32_t frame) const {
    const double t = double(frame) / fps - titleSeconds;
    if (morphSeconds <= 0.0f) {
        return t >= 0.0 ? 1.0f : 0.0f;
    }
    return float(std::clamp(t / morphSeconds, 0.0, 1.0));
}

double MorphTimeline::totalAngle() const { return 2.0 * 3.14159265358979323846 * turns; }

double MorphTimeline::angleAt(uint32_t frame) const {
    const double t = double(frame) / fps - titleSeconds;
    if (t <= 0.0) {
        return 0.0;
    }
    // Constant acceleration during the morph reaches the speed `rate` and
    // covers rate * morph / 2; the spin adds rate * spin.
    const double morph = morphSeconds;
    const double rate = totalAngle() / (0.5 * morph + spinSeconds);
    if (t < morph) {
        return rate * t * t / (2.0 * morph);
    }
    return rate * (0.5 * morph + (t - morph));
}

TextMorphStart prepareTextMorph(std::vector<Gaussian3D> text, const std::vector<Gaussian3D>& scene,
                                const glm::mat4& view, const glm::vec3& centre) {
    if (text.size() != scene.size() || text.empty()) {
        throw std::invalid_argument("prepareTextMorph: need as many text Gaussians as the scene has");
    }
    float minX = text[0].position[0], maxX = minX;
    for (const Gaussian3D& g : text) {
        minX = std::min(minX, g.position[0]);
        maxX = std::max(maxX, g.position[0]);
    }
    const float width = std::max(maxX - minX, 1e-12f);
    std::vector<float> delays(text.size());
    for (size_t i = 0; i < text.size(); ++i) {
        delays[i] = (text[i].position[0] - minX) / width;
    }

    placeFacingCamera(text, view, centre);
    const std::vector<uint32_t> match = matchByScreenPosition(text, scene, view);
    TextMorphStart start;
    start.gaussians.resize(scene.size());
    start.delays.resize(scene.size());
    for (size_t i = 0; i < scene.size(); ++i) {
        start.gaussians[i] = text[match[i]];
        start.delays[i] = delays[match[i]];
    }
    return start;
}

} // namespace klartraum
