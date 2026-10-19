// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_TREE_TREE_RENDER_INPUTS_HPP
#define KLARTRAUM_TREE_TREE_RENDER_INPUTS_HPP

#include "klartraum/vulkan_gaussian_splatting_types.hpp"

namespace klartraum {

/// Optional buffers the classical tree renderers read; leave an element empty when not used.
struct TreeRenderInputs {
    /// Depth of Gaussians drawn before (GsplatConfig::outputDepth); fragments behind it are hidden when
    /// TreeRenderSettings::occlusion.x is 1 (tree_occlusion.glsl).
    BufferRef sceneDepth;
    /// The light volume of a TreeLightVolume (tree_light.glsl), used when TreeRenderSettings::lightGridSize.w is 1.
    BufferRef light;
};

} // namespace klartraum

#endif // KLARTRAUM_TREE_TREE_RENDER_INPUTS_HPP
