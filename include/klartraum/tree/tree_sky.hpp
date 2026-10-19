// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_TREE_TREE_SKY_HPP
#define KLARTRAUM_TREE_TREE_SKY_HPP

#include <glm/glm.hpp>

#include "klartraum/tree/tree_model.hpp"

namespace klartraum {

/// The light of the sky at a time of day and season.
struct TreeSky {
    glm::vec3 sunDirection{0.0f, 1.0f, 0.0f};  ///< unit vector towards the sun; +x east, +y up, -z north
    float sunElevation = 0.0f;                 ///< radians above the horizon
    glm::vec3 moonDirection{0.0f, 1.0f, 0.0f}; ///< towards the full moon, opposite the sun
    float moonElevation = 0.0f;
    glm::vec3 lightDirection{0.0f, 1.0f, 0.0f}; ///< the light that casts shadows: the sun by day, the moon by night
    float lightIntensity = 0.0f;
    glm::vec3 lightColor{1.0f};
    glm::vec3 sky{0.0f};          ///< ambient light from above
    glm::vec3 ground{0.0f};       ///< ambient light from below
    float exposure = 1.4f;        ///< higher at night, like eyes that adapt
    float daylight = 0.0f;        ///< 0 at night, 1 by day, in between at dusk and dawn
    float lanternGlow = 0.0f;     ///< how strongly the lantern glows: from dusk to dawn
    glm::vec3 capturedTint{1.0f}; ///< for the captured lantern, which was lit by daylight
};

/**
 * @brief The sky at `timeOfDay` (0 midnight, 0.5 noon) in season S (as TreeFrame), at `latitude` degrees north.
 *
 * The sun's declination follows the season (0 at the equinoxes S = 1 and 3,
 * +23.44 degrees at the summer solstice S = 2, -23.44 at the winter solstice S
 * = 0), so in winter the sun stays lower and the days are shorter. The light
 * that casts shadows is the sun while it is up and the full moon at night; the
 * moon rises in brightness only once the sun is down, so the switch between
 * them happens while both are dark.
 */
TreeSky treeSkyAt(float timeOfDay, float season, float latitude = 48.0f);

/// Writes the light of `sky` into `settings`; the lantern's flame at `flame` flickers with `time` in seconds.
void applyTreeSky(TreeRenderSettings& settings, const TreeSky& sky, glm::vec3 flame, float time);

/**
 * @brief The days of the bonsai video: one season per day, changing while it is dark.
 *
 * Day d (d = 0 .. days - 1) shows season seasons[d % 4] from before sunrise
 * through the night to before the next sunrise; around midnight (time of day
 * 0.90 to 1.10) the season changes smoothly to the next day's (after the last
 * day back to the first, so the video loops).
 */
struct BonsaiDays {
    float secondsPerDay = 10.0f;
    uint32_t days = 4;
    float seasons[4] = {1.25f, 2.0f, 3.25f, 4.0f}; ///< spring, summer, fall, winter
    float dawn = 0.12f;                            ///< time of day each day starts at: before sunrise, also in summer

    float duration() const { return secondsPerDay * float(days); }
    uint32_t dayAt(float seconds) const;
    float timeOfDayAt(float seconds) const;
    float seasonAt(float seconds) const;
};

} // namespace klartraum

#endif // KLARTRAUM_TREE_TREE_SKY_HPP
