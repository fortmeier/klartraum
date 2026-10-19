// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Hides classical fragments behind Gaussians drawn before into the same
// target, by the depth buffer of a splatting backend (GsplatConfig::outputDepth:
// per pixel the NDC depth, +inf where there is no surface). The includer
// defines TREE_OCCLUSION_BINDING (set 1).

layout(std430, set = 1, binding = TREE_OCCLUSION_BINDING) readonly buffer SceneDepth {
    float sceneDepth[];
};

bool treeBehindScene(TreeRenderSettings s) {
    if (s.occlusion.x == 0.0f) {
        return false;
    }
    ivec2 p = ivec2(gl_FragCoord.xy);
    return gl_FragCoord.z > sceneDepth[p.y * int(s.viewport.x) + p.x];
}
