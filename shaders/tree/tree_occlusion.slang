// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import glsl;

// Hides classical fragments behind Gaussians drawn before into the same
// target, by the depth buffer of a splatting backend (GsplatConfig::outputDepth:
// per pixel the NDC depth, +inf where there is no surface). The includer
// defines TREE_OCCLUSION_BINDING (set 1).

[[vk::binding(TREE_OCCLUSION_BINDING, 1)]] StructuredBuffer<float> sceneDepth;

bool treeBehindScene(TreeRenderSettings s) {
    if (s.occlusion.x == 0.0f) {
        return false;
    }
    ivec2 p = ivec2(gl_FragCoord.xy);
    return gl_FragCoord.z > sceneDepth[p.y * int(s.viewport.x) + p.x];
}
