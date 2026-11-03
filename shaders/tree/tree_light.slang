// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import glsl;

// Reads the light volume of TreeLightVolume: per voxel the sun transmittance
// and the sky visibility, interpolated trilinearly between voxel centres. The
// includer defines TREE_LIGHT_SET and TREE_LIGHT_BINDING.

[[vk::binding(TREE_LIGHT_BINDING, TREE_LIGHT_SET)]] StructuredBuffer<vec4> treeLight; // xy through everything, zw through the trees only

// The light volume at p, interpolated: (sun transmittance, sky visibility)
// through all occluders in xy, through the trees only in zw; (1, 1, 1, 1)
// without a light volume. Points outside the grid take the value of the
// nearest voxel.
vec4 treeLightAll(TreeRenderSettings s, vec3 p) {
    if (s.lightGridSize.w == 0u) {
        return vec4(1.0f);
    }
    ivec3 size = ivec3(s.lightGridSize.xyz);
    vec3 g = clamp((p - s.lightGridMin.xyz) / s.lightGridMin.w - 0.5f, vec3(0.0f), vec3(size - 1));
    ivec3 i0 = ivec3(floor(g));
    ivec3 i1 = min(i0 + 1, size - 1);
    vec3 f = g - vec3(i0);
    uvec4 n = s.lightGridSize;
    vec4 c000 = treeLight[treeLightIndex(i0.x, i0.y, i0.z, n)];
    vec4 c100 = treeLight[treeLightIndex(i1.x, i0.y, i0.z, n)];
    vec4 c010 = treeLight[treeLightIndex(i0.x, i1.y, i0.z, n)];
    vec4 c110 = treeLight[treeLightIndex(i1.x, i1.y, i0.z, n)];
    vec4 c001 = treeLight[treeLightIndex(i0.x, i0.y, i1.z, n)];
    vec4 c101 = treeLight[treeLightIndex(i1.x, i0.y, i1.z, n)];
    vec4 c011 = treeLight[treeLightIndex(i0.x, i1.y, i1.z, n)];
    vec4 c111 = treeLight[treeLightIndex(i1.x, i1.y, i1.z, n)];
    return mix(mix(mix(c000, c100, f.x), mix(c010, c110, f.x), f.y), mix(mix(c001, c101, f.x), mix(c011, c111, f.x), f.y),
               f.z);
}

// (sun transmittance, sky visibility) at p through all occluders.
vec2 treeLightAt(TreeRenderSettings s, vec3 p) {
    return treeLightAll(s, p).xy;
}

// (sun transmittance, sky visibility) at p through the trees only: for
// captured Gaussians, whose colours already contain their own occlusion.
vec2 treeLightOfTreesAt(TreeRenderSettings s, vec3 p) {
    return treeLightAll(s, p).zw;
}

// (sun transmittance, sky visibility) of a captured object's point p, whose colours hold the light of the capture
// but not that of the scene. The sun is read a voxel and a half towards the light, where a point facing the light is
// outside its own surface and a point facing away is inside the object (no normal needed, captured Gaussians have
// none that points outwards). The trees' shadows count fully; the shadows of the static occluders, the object itself
// included, at three quarters, as the captured colours already hold some of the object's own shading. The sky is
// seen through the trees only.
vec2 treeCapturedLightAt(TreeRenderSettings s, vec3 p) {
    vec4 towardsLight = treeLightAll(s, p + normalize(s.sun.xyz) * (1.5f * s.lightGridMin.w));
    float statics = towardsLight.x / max(towardsLight.z, 1e-4f);
    return vec2(towardsLight.z * mix(1.0f, statics, 0.75f), treeLightAll(s, p).w);
}
