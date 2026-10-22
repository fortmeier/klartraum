// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Colour output of the classical tree renderer; the lighting is treeShade() in tree_shared.hpp. The includer
// includes tree_render_common.glsl (the camera) before.

vec4 treeOutputColor(vec3 linearColor, TreeRenderSettings s) {
    return vec4(treeDisplayColor(linearColor, s.sky.w), 1.0f);
}

// What a surface of the trees at pos writes: its displayed colour through the haze (treeAtmosphere()), or while
// baking impostors (TreeImpostorAtlas, s.occlusion.y) 1 its albedo (square-rooted for precision, at least 1/255 so
// that it stands out from the black background) or 2 its normal (mapped to [0, 1]).
vec4 treeSurfaceOutput(vec3 linearColor, vec3 albedo, vec3 normal, vec3 pos, TreeRenderSettings s) {
    if (s.occlusion.y == 1.0f) {
        return vec4(max(sqrt(clamp(albedo, vec3(0.0f), vec3(1.0f))), vec3(1.0f / 255.0f)), 1.0f);
    }
    if (s.occlusion.y == 2.0f) {
        return vec4(0.5f * normal + vec3(0.5f), 1.0f);
    }
    return treeOutputColor(treeAtmosphere(linearColor, pos, camera.cameraWorldPos.xyz, s), s);
}
