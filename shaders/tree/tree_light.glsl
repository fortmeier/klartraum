// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Reads the light volume of TreeLightVolume: per voxel the sun transmittance
// and the sky visibility, interpolated trilinearly between voxel centres. The
// includer defines TREE_LIGHT_SET and TREE_LIGHT_BINDING.

layout(std430, set = TREE_LIGHT_SET, binding = TREE_LIGHT_BINDING) readonly buffer TreeLightGrid {
    vec2 treeLight[];
};

// (sun transmittance, sky visibility) at p, (1, 1) without a light volume.
// Points outside the grid take the value of the nearest voxel.
vec2 treeLightAt(TreeRenderSettings s, vec3 p) {
    if (s.lightGridSize.w == 0u) {
        return vec2(1.0f);
    }
    ivec3 size = ivec3(s.lightGridSize.xyz);
    vec3 g = clamp((p - s.lightGridMin.xyz) / s.lightGridMin.w - 0.5f, vec3(0.0f), vec3(size - 1));
    ivec3 i0 = ivec3(floor(g));
    ivec3 i1 = min(i0 + 1, size - 1);
    vec3 f = g - vec3(i0);
    uvec4 n = s.lightGridSize;
    vec2 c000 = treeLight[treeLightIndex(i0.x, i0.y, i0.z, n)];
    vec2 c100 = treeLight[treeLightIndex(i1.x, i0.y, i0.z, n)];
    vec2 c010 = treeLight[treeLightIndex(i0.x, i1.y, i0.z, n)];
    vec2 c110 = treeLight[treeLightIndex(i1.x, i1.y, i0.z, n)];
    vec2 c001 = treeLight[treeLightIndex(i0.x, i0.y, i1.z, n)];
    vec2 c101 = treeLight[treeLightIndex(i1.x, i0.y, i1.z, n)];
    vec2 c011 = treeLight[treeLightIndex(i0.x, i1.y, i1.z, n)];
    vec2 c111 = treeLight[treeLightIndex(i1.x, i1.y, i1.z, n)];
    return mix(mix(mix(c000, c100, f.x), mix(c010, c110, f.x), f.y), mix(mix(c001, c101, f.x), mix(c011, c111, f.x), f.y),
               f.z);
}
