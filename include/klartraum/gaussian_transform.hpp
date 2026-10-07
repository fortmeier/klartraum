// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_GAUSSIAN_TRANSFORM_HPP
#define KLARTRAUM_GAUSSIAN_TRANSFORM_HPP

#include <array>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include "klartraum/vulkan_gaussian_splatting_types.hpp"

namespace klartraum {

// Moves Gaussians by a similarity transform: each Gaussian is scaled by
// `scale` about the origin, rotated by `rotation` and then moved by
// `translation`. Orientations and the view-dependent colour (spherical
// harmonics of degree 1 to 3) turn with the scene, so it looks the same from a
// camera that is moved along. `scale` must be positive.
void transformGaussians(std::vector<Gaussian3D>& gaussians, const glm::quat& rotation, float scale = 1.0f,
                        const glm::vec3& translation = glm::vec3(0.0f));

// The spherical-harmonics basis of degree 1 to 3 as the splatting shaders
// evaluate it (computeSH in shaders/gsplat): the 15 values the coefficients in
// Gaussian3D::shR/shG/shB are weighted with for the unit direction `dir`.
std::array<float, 15> shBasis(const glm::vec3& dir);

} // namespace klartraum

#endif // KLARTRAUM_GAUSSIAN_TRANSFORM_HPP
