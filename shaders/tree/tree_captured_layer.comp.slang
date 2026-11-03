// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import glsl;

// Draws a layer of premultiplied colours with coverage (a MeshMorph's antialiased mesh) over an image, lit like a
// captured object of the tree renderers: each pixel is shaded by the light volume at the middle of the triangle it
// shows (treeCapturedShadeAt()) and tinted by treeCapturedColor(). Pixels at the outline that show no triangle take
// the light of a neighbour that does.
// image = lit(layer.rgb) + (1 - layer.a) * image.

#define TREE_FN
#include "klartraum/tree/tree_shared.hpp"
#define TREE_LIGHT_SET 0
#define TREE_LIGHT_BINDING 7
#include "tree_light.slang"

struct Push {
    uint width;
    uint height;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

[[vk::binding(0, 0)]] [[vk::image_format("rgba8")]] RWTexture2D<vec4> image;
[[vk::binding(1, 0)]] StructuredBuffer<vec4> layer;      // per pixel, row-major
[[vk::binding(2, 0)]] StructuredBuffer<uint> visibility; // per pixel: triangle + 1, 0 for none
[[vk::binding(3, 0)]] StructuredBuffer<vec4> positions;  // per vertex, in the morph's frame
[[vk::binding(4, 0)]] StructuredBuffer<uint> triangles;  // three vertices each
[[vk::binding(5, 0)]] StructuredBuffer<mat4> worldFromMorph;
[[vk::binding(6, 0)]] StructuredBuffer<TreeRenderSettings> settings;

[shader("compute")]
[numthreads(16, 16, 1)]
void main() {
    uvec2 p = gl_GlobalInvocationID.xy;
    if (p.x >= pc.width || p.y >= pc.height) {
        return;
    }
    vec4 top = layer[p.y * pc.width + p.x];
    if (top.a <= 0.0f) {
        return;
    }
    uint id = visibility[p.y * pc.width + p.x];
    for (int k = 0; k < 9 && id == 0u; ++k) {
        ivec2 q = ivec2(p) + ivec2(k % 3 - 1, k / 3 - 1);
        if (all(greaterThanEqual(q, ivec2(0))) && all(lessThan(q, ivec2(int(pc.width), int(pc.height))))) {
            id = visibility[uint(q.y) * pc.width + uint(q.x)];
        }
    }
    vec3 colour = top.rgb / top.a;
    if (id != 0u) {
        mat4 m = worldFromMorph[0];
        uint t = id - 1u;
        vec3 a = (m * vec4(positions[triangles[3u * t]].xyz, 1.0f)).xyz;
        vec3 b = (m * vec4(positions[triangles[3u * t + 1u]].xyz, 1.0f)).xyz;
        vec3 c = (m * vec4(positions[triangles[3u * t + 2u]].xyz, 1.0f)).xyz;
        vec3 middle = (a + b + c) / 3.0f;
        // The face normal, turned away from the mesh's origin.
        vec3 normal = normalize(cross(b - a, c - a));
        if (dot(normal, middle - vec3(m[3].x, m[3].y, m[3].z)) < 0.0f) {
            normal = -normal;
        }
        TreeRenderSettings s = settings[0];
        float shade = treeCapturedShadeAt(s, middle, normal);
        colour = treeCapturedColor(colour, normal, middle, shade, s);
    }
    vec4 below = image[p];
    image[p] = vec4(clamp(colour * top.a + (1.0f - top.a) * below.rgb, 0.0f, 1.0f), 1.0f);
}
