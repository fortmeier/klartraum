// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import glsl;

// A blade of a visible clump of grass, for its classical strip
// (tree_grass.vert) and its Gaussian (tree_grass_gaussians.comp).

struct GrassBlade {
    vec3 foot;   // on the ground's slope within the clump's cell
    float height;
    vec2 facing; // the horizontal direction it bends to
    vec2 bend;   // horizontal offset of its tip per unit of height (lean and wind)
};

// Blade `blade` of `clump`: its foot within the clump's cell, its height (shorter under snow), bent towards its
// facing, more for longer blades, and swayed by `wind` (x time in seconds, y strength, zw the direction).
GrassBlade treeGrassBlade(TreeGrassVisible clump, TreeGrassParams g, uint blade, vec4 wind) {
    GrassBlade b;
    uint key = treeHash2(floatBitsToUint(clump.kind.w), blade);
    b.height = clump.posHeight.w * (0.6f + 0.7f * treeRand(key, 1u)) * (1.0f - 0.85f * clump.ground.z);
    float a = 2.0f * TREE_PI * treeRand(key, 2u);
    vec2 o = 0.5f * g.grid.x * sqrt(treeRand(key, 3u)) * vec2(cos(a), sin(a));
    b.foot = clump.posHeight.xyz + vec3(o.x, clump.ground.x * o.x + clump.ground.y * o.y, o.y);
    float facing = clump.kind.y + 2.0f * TREE_PI * treeRand(key, 4u);
    b.facing = vec2(cos(facing), sin(facing));
    float lean = 0.15f + 0.45f * treeRand(key, 5u);
    float phase = 2.0f * wind.x + 0.35f * dot(b.foot.xz * g.grid.w, wind.zw) + 6.0f * treeRand(key, 6u);
    b.bend = lean * b.facing + wind.y * (0.35f + 0.25f * sin(phase)) * wind.zw;
    return b;
}

// The point at t in [0, 1] along the blade's middle, and its tangent.
vec3 treeGrassBladePoint(GrassBlade b, float t) {
    return b.foot +
           b.height * vec3(b.bend.x * t * t, t * (1.0f - 0.25f * dot(b.bend, b.bend) * t), b.bend.y * t * t);
}

vec3 treeGrassBladeTangent(GrassBlade b, float t) {
    return normalize(vec3(2.0f * b.bend.x * t, 1.0f, 2.0f * b.bend.y * t));
}
