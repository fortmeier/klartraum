// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/demo_sky_marks.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>

namespace klartraum {

namespace {

constexpr float kPi = 3.14159265f;

// Blends `colour` into the image where `inside` covers it (4 x 4 samples per pixel), within `reach` pixels of
// `centre`, over the pixels that show the background grey only.
void drawMark(std::vector<uint8_t>& bgra, uint32_t width, uint32_t height, glm::vec2 centre, float reach,
              glm::vec3 colour, float opacity, float background, const std::function<bool(glm::vec2)>& inside) {
    const int grey = int(std::lround(std::clamp(background, 0.0f, 1.0f) * 255.0f));
    const int x0 = std::max(0, int(std::floor(centre.x - reach))), x1 = std::min(int(width) - 1, int(centre.x + reach));
    const int y0 = std::max(0, int(std::floor(centre.y - reach))),
              y1 = std::min(int(height) - 1, int(centre.y + reach));
    for (int y = y0; y <= y1; ++y)
        for (int x = x0; x <= x1; ++x) {
            uint8_t* pixel = &bgra[4 * (size_t(y) * width + size_t(x))];
            if (std::abs(int(pixel[0]) - grey) > 3 || std::abs(int(pixel[1]) - grey) > 3 ||
                std::abs(int(pixel[2]) - grey) > 3) {
                continue;
            }
            int covered = 0;
            for (int s = 0; s < 16; ++s) {
                const glm::vec2 sample(float(x) + (float(s % 4) + 0.5f) / 4.0f,
                                       float(y) + (float(s / 4) + 0.5f) / 4.0f);
                covered += inside(sample - centre) ? 1 : 0;
            }
            const float weight = opacity * float(covered) / 16.0f;
            const glm::vec3 bgr(colour.b, colour.g, colour.r);
            for (int c = 0; c < 3; ++c) {
                const float value = float(pixel[c]) + (255.0f * bgr[c] - float(pixel[c])) * weight;
                pixel[c] = uint8_t(std::lround(std::clamp(value, 0.0f, 255.0f)));
            }
        }
}

} // namespace

SkyMark skyMarkAt(const glm::mat4& view, glm::vec3 direction, uint32_t width, uint32_t height, float halfWidthAngle,
                  float horizon, float zenith) {
    SkyMark mark;
    const glm::vec3 d = glm::normalize(direction);
    const float elevation = std::asin(std::clamp(d.y, -1.0f, 1.0f));
    // The camera's forward and right directions in the horizontal plane.
    const glm::vec3 forward = -glm::vec3(view[0][2], view[1][2], view[2][2]);
    const glm::vec2 ahead = glm::normalize(glm::vec2(forward.x, forward.z));
    const glm::vec2 right(-ahead.y, ahead.x);
    const glm::vec2 flat(d.x, d.z);
    if (elevation <= 0.0f || glm::length(flat) < 1e-6f) {
        // Below the horizon, or straight up, where it has no side.
        mark.visible = elevation > 0.0f;
        mark.pixel = glm::vec2(0.5f * float(width), zenith * float(height));
        return mark;
    }
    const float angle = std::atan2(glm::dot(flat, right), glm::dot(flat, ahead));
    if (std::abs(angle) >= halfWidthAngle) {
        return mark;
    }
    mark.visible = true;
    mark.pixel = glm::vec2(0.5f * float(width) * (1.0f + angle / halfWidthAngle),
                           float(height) * (horizon + (zenith - horizon) * elevation / (0.5f * kPi)));
    return mark;
}

void drawSunMark(std::vector<uint8_t>& bgra, uint32_t width, uint32_t height, glm::vec2 centre, float radius,
                 glm::vec3 colour, float opacity, float background) {
    drawMark(bgra, width, height, centre, 2.1f * radius, colour, opacity, background, [radius](glm::vec2 q) {
        const float r = glm::length(q);
        if (r < radius) {
            return true;
        }
        if (r < 1.35f * radius || r > 2.0f * radius) {
            return false;
        }
        // Eight rays, one every 45 degrees from the right.
        const float angle = std::atan2(q.y, q.x);
        const float nearest = std::round(angle / (0.25f * kPi)) * (0.25f * kPi);
        return std::abs(r * std::sin(angle - nearest)) < 0.13f * radius;
    });
}

void drawMoonMark(std::vector<uint8_t>& bgra, uint32_t width, uint32_t height, glm::vec2 centre, float radius,
                  glm::vec2 lit, glm::vec3 colour, float opacity, float background) {
    // The moon's disc without a disc shifted away from its lit side.
    const glm::vec2 cut = -0.5f * radius * glm::normalize(lit);
    drawMark(bgra, width, height, centre, 1.1f * radius, colour, opacity, background,
             [radius, cut](glm::vec2 q) { return glm::length(q) < radius && glm::length(q - cut) > 0.92f * radius; });
}

} // namespace klartraum
