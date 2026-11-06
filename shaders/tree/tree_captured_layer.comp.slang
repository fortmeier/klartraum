// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import glsl;

// Draws a layer of premultiplied colours with coverage (a MeshMorph's antialiased mesh) over an image, lit like a
// captured object of the tree renderers: each pixel is shaded by the light volume where its ray meets the triangle it
// shows (treeCapturedShadeAt()), so that the light changes smoothly across the mesh, and tinted by
// treeCapturedColor(). Pixels at the outline that show no triangle take a neighbour's triangle.
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
[[vk::binding(5, 0)]] StructuredBuffer<mat4> drawing; // world from the morph's frame, the morph's frame from clip
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
        mat4 m = drawing[0];
        uint t = id - 1u;
        vec3 a = positions[triangles[3u * t]].xyz;
        vec3 b = positions[triangles[3u * t + 1u]].xyz;
        vec3 c = positions[triangles[3u * t + 2u]].xyz;
        // The pixel's ray in the morph's frame, met with the triangle's plane.
        vec2 ndc = (vec2(p) + 0.5f) / vec2(float(pc.width), float(pc.height)) * 2.0f - 1.0f;
        vec4 near = drawing[1] * vec4(ndc.x, ndc.y, 0.0f, 1.0f);
        vec4 far = drawing[1] * vec4(ndc.x, ndc.y, 1.0f, 1.0f);
        vec3 from = near.xyz / near.w;
        vec3 along = far.xyz / far.w - from;
        vec3 normal = cross(b - a, c - a);
        float across = dot(normal, along);
        vec3 hit = (a + b + c) / 3.0f;
        if (abs(across) > 1e-12f) {
            hit = from + along * (dot(normal, a - from) / across);
        }
        vec3 world = (m * vec4(hit, 1.0f)).xyz;
        // The face normal, turned away from the mesh's origin, for snow only: a face normal would light the mesh
        // in facets.
        vec3 face = normalize((m * vec4(normal, 0.0f)).xyz);
        if (dot(face, world - vec3(m[3].x, m[3].y, m[3].z)) < 0.0f) {
            face = -face;
        }
        TreeRenderSettings s = settings[0];
        if (treeWritesGBuffer(s)) {
            // The ray's start on the near plane stands in for the eye.
            colour = treeGBufferColor(s, TREE_CLASS_OBJECT, world, (m * vec4(from, 1.0f)).xyz);
        } else {
            float shade = treeCapturedShadeAt(s, world, vec3(0.0f));
            colour = treeCapturedColor(colour, face, world, shade, s);
        }
    }
    vec4 below = image[p];
    image[p] = vec4(clamp(colour * top.a + (1.0f - top.a) * below.rgb, 0.0f, 1.0f), 1.0f);
}
