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

// How much of the light reaches the point p of a captured object, whose colours hold the light of the capture but
// not that of the scene (1 in full sun under an open sky, see treeCapturedShade()). The sun is read a voxel and a
// half towards the light, where a point facing the light is outside its own surface and a point facing away is
// inside the object, through the trees and the static occluders (the object itself included) alike. A surface whose
// `facing` (the side that is seen, its length how far that is known: 1 for a surface, 0 for unknown) turns away from
// the sun gets no sun either, which also holds for thin walls, where the point a voxel and a half on lies beyond the
// wall. Where the object or the
// other static occluders shut out the sun, the shade is lifted by a quarter of the sun's share, the same everywhere,
// as the captured colours already hold some of the object's own shading; in the object's own shadow, the trees thus
// have no sun left to block. The sky is seen through the trees over the part the static occluders leave open.
float treeCapturedShadeAt(TreeRenderSettings s, vec3 p, vec3 facing) {
    vec3 sun = normalize(s.sun.xyz);
    vec4 towardsLight = treeLightAll(s, p + sun * (1.5f * s.lightGridMin.w));
    float statics = clamp(towardsLight.x / max(towardsLight.z, 1e-4f), 0.0f, 1.0f);
    float known = length(facing);
    if (known > 0.0f) {
        statics *= mix(1.0f, smoothstep(-0.05f, 0.2f, dot(facing / known, sun)), min(known, 1.0f));
    }
    float shade = treeCapturedShade(s, vec2(towardsLight.z * statics, treeLightAll(s, p).w));
    float lit = treeCapturedShade(s, vec2(1.0f, 0.0f));
    return shade + 0.25f * lit * (1.0f - statics);
}
