// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_DEMO_SKY_MARKS_HPP
#define KLARTRAUM_DEMO_SKY_MARKS_HPP

#include <cstdint>
#include <vector>

#include <glm/glm.hpp>

namespace klartraum {

/// Where a sun or moon symbol stands in a demo frame.
struct SkyMark {
    bool visible = false;
    glm::vec2 pixel{0.0f}; ///< its centre, in pixels from the frame's top left
};

/// Where the symbol of a light in `direction` (world, towards the light) stands in a frame of width x height seen
/// with `view` (camera from world): horizontally at the light's angle from the camera's forward direction, the frame's
/// sides at +-`halfWidthAngle` radians; vertically by the light's elevation above the horizon, from `horizon` (a
/// fraction of the height from the top) at 0 up to `zenith` at a right angle. A camera looking down at a scene thus
/// still shows the light on the side it shines from. Hidden beyond the frame's sides and below the horizon.
SkyMark skyMarkAt(const glm::mat4& view, glm::vec3 direction, uint32_t width, uint32_t height,
                  float halfWidthAngle = 1.2f, float horizon = 0.5f, float zenith = 0.08f);

/// Draws a sun, a disc of `radius` pixels with eight rays, in `colour` with `opacity` into the tightly packed BGRA
/// image, antialiased, only over pixels of the background grey `background` (in [0, 1]), so that it stands behind
/// whatever is drawn in front of the background.
void drawSunMark(std::vector<uint8_t>& bgra, uint32_t width, uint32_t height, glm::vec2 centre, float radius,
                 glm::vec3 colour, float opacity, float background);

/// Draws a crescent moon of `radius` pixels whose lit side faces `lit` (a direction in the image, y down), like
/// drawSunMark().
void drawMoonMark(std::vector<uint8_t>& bgra, uint32_t width, uint32_t height, glm::vec2 centre, float radius,
                  glm::vec2 lit, glm::vec3 colour, float opacity, float background);

} // namespace klartraum

#endif // KLARTRAUM_DEMO_SKY_MARKS_HPP
