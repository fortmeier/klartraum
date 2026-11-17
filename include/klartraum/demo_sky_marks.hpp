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
    float opacity = 0.0f;  ///< 1, fading to 0 as the light goes round behind the camera
};

/// Where the symbol of a light in `direction` (world, towards the light) stands in a frame of width x height seen
/// with `view` (camera from world), as on a view of the sky dome from below: horizontally by how far the light lies
/// to the camera's right, the frame's sides at `sideReach`; vertically by the sine of its elevation, from `horizon`
/// (a fraction of the height from the top) at the horizon up to `zenith` straight up. A camera looking down at a
/// scene thus still shows the light on the side it shines from, and a light passing overhead moves smoothly. Hidden
/// beyond the frame's sides, behind the camera and below the horizon.
SkyMark skyMarkAt(const glm::mat4& view, glm::vec3 direction, uint32_t width, uint32_t height, float sideReach = 0.9f,
                  float horizon = 0.42f, float zenith = 0.06f);

/// The direction in the frame (y down) in which the lit side of the moon in `moon` faces the sun in `sun` (both
/// world directions) for a camera with `view`: towards the sun as projected onto the camera's image plane. Unit length.
glm::vec2 litSideTowards(const glm::mat4& view, glm::vec3 moon, glm::vec3 sun);

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
