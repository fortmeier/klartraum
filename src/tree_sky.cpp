// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/tree/tree_sky.hpp"

#include <algorithm>
#include <cmath>

namespace klartraum {

namespace {

constexpr float kPi = 3.14159265f;

float smoothstep(float edge0, float edge1, float x) {
    const float t = std::clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// Direction towards a body at declination `declination` and hour angle `hour`
// (radians, 0 at its highest), seen from `latitude`: +x east, +y up, -z north.
glm::vec3 celestialDirection(float declination, float hour, float latitude) {
    const float east = -std::cos(declination) * std::sin(hour);
    const float north =
        std::sin(declination) * std::cos(latitude) - std::cos(declination) * std::sin(latitude) * std::cos(hour);
    const float up =
        std::sin(latitude) * std::sin(declination) + std::cos(latitude) * std::cos(declination) * std::cos(hour);
    return glm::normalize(glm::vec3(east, up, -north));
}

} // namespace

TreeSky treeSkyAt(float timeOfDay, float season, float latitude) {
    TreeSky sky;
    const float phi = glm::radians(latitude);
    const float declination = glm::radians(23.44f) * std::sin((season - 1.0f) * 0.5f * kPi);
    const float hour = 2.0f * kPi * (timeOfDay - 0.5f);
    sky.sunDirection = celestialDirection(declination, hour, phi);
    sky.moonDirection = celestialDirection(-declination, hour + kPi, phi);
    sky.sunElevation = std::asin(std::clamp(sky.sunDirection.y, -1.0f, 1.0f));
    sky.moonElevation = std::asin(std::clamp(sky.moonDirection.y, -1.0f, 1.0f));

    const float sunUp = sky.sunDirection.y;
    sky.daylight = smoothstep(-0.1f, 0.25f, sunUp);
    // Rising over the first 16 degrees above the horizon, so the shadows fade in over half a second of a
    // 10 s day rather than appear.
    const float sunIntensity = 1.6f * smoothstep(-0.02f, 0.28f, sunUp);
    const float moonIntensity =
        0.15f * smoothstep(-0.02f, 0.15f, sky.moonDirection.y) * (1.0f - smoothstep(-0.12f, -0.02f, sunUp));
    if (sunIntensity >= moonIntensity) {
        sky.lightDirection = sky.sunDirection;
        sky.lightIntensity = sunIntensity;
        sky.lightColor =
            glm::mix(glm::vec3(1.0f, 0.42f, 0.18f), glm::vec3(1.0f, 0.96f, 0.9f), smoothstep(0.0f, 0.35f, sunUp));
    } else {
        sky.lightDirection = sky.moonDirection;
        sky.lightIntensity = moonIntensity;
        sky.lightColor = glm::vec3(0.55f, 0.68f, 1.0f);
    }

    // Twilight reddens the sky around sunrise and sunset.
    const float dusk = 4.0f * sky.daylight * (1.0f - sky.daylight);
    const glm::vec3 night(0.012f, 0.016f, 0.035f);
    const glm::vec3 day(0.32f, 0.36f, 0.42f);
    sky.sky = glm::mix(night, day, sky.daylight) + dusk * glm::vec3(0.12f, 0.05f, 0.04f);
    sky.ground = glm::mix(glm::vec3(0.006f, 0.006f, 0.01f), glm::vec3(0.16f, 0.13f, 0.1f), sky.daylight);
    // The eyes adapt more slowly than the light changes at dawn and dusk.
    sky.exposure = glm::mix(3.0f, 1.4f, smoothstep(-0.3f, 0.4f, sunUp));
    sky.lanternGlow = 1.0f - smoothstep(0.0f, 0.35f, sky.daylight);
    // The capture shows the lantern in daylight; at night it is moonlit, and its flame glows warm.
    const glm::vec3 moonlit(0.1f, 0.12f, 0.2f);
    sky.capturedTint =
        glm::mix(moonlit, glm::vec3(1.0f), sky.daylight) + sky.lanternGlow * glm::vec3(0.55f, 0.3f, 0.09f);
    return sky;
}

void applyTreeSky(TreeRenderSettings& settings, const TreeSky& sky, glm::vec3 flame, float time) {
    settings.sun = glm::vec4(sky.lightDirection, sky.lightIntensity);
    settings.lightColor = glm::vec4(sky.lightColor, 0.0f);
    settings.sky = glm::vec4(sky.sky, sky.exposure);
    settings.ground = glm::vec4(sky.ground, 0.0f);
    const float flicker = 0.85f + 0.15f * tree::treeValueNoise(glm::vec3(8.0f * time, 0.0f, 0.0f));
    settings.pointLight = glm::vec4(flame, 0.0f);
    settings.pointColor = glm::vec4(glm::vec3(1.0f, 0.55f, 0.2f) * 0.035f * sky.lanternGlow * flicker, 0.0f);
    settings.capturedTint = glm::vec4(sky.capturedTint, 0.0f);
}

uint32_t BonsaiDays::dayAt(float seconds) const {
    return std::min(uint32_t(std::max(seconds, 0.0f) / secondsPerDay), days - 1);
}

float BonsaiDays::timeOfDayAt(float seconds) const {
    const float f = seconds / secondsPerDay - float(dayAt(seconds));
    const float t = dawn + f;
    return t - std::floor(t);
}

float BonsaiDays::seasonAt(float seconds) const {
    // The season of day d, counted on so that every change runs forwards; after the last day, the first again.
    const auto seasonOf = [this](uint32_t d) {
        return d < days ? seasons[d % 4] + 4.0f * float(d / 4) : seasons[0] + 4.0f * float((days + 3) / 4);
    };
    // Days since the first midnight, which ends day 0: midnight m, between day m and m + 1, lies at x = m. The
    // season changes over a fifth of a day around the nearest midnight, whichever day the midnight falls in.
    const float x = std::max(seconds, 0.0f) / secondsPerDay - (1.0f - dawn);
    const float nearest = std::round(x);
    if (nearest < 0.0f) {
        return tree::treeSeasonWrap(seasonOf(0));
    }
    const uint32_t m = std::min(uint32_t(nearest), days - 1);
    const float change = smoothstep(-0.1f, 0.1f, x - float(m));
    return tree::treeSeasonWrap(seasonOf(m) + (seasonOf(m + 1) - seasonOf(m)) * change);
}

} // namespace klartraum
