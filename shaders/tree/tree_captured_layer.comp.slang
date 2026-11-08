// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import glsl;

// Draws a layer of premultiplied colours with coverage (a MeshMorph's antialiased mesh) over an image, lit like a
// captured object of the tree renderers: each pixel is shaded by the light volume where its ray meets the triangle it
// shows (treeCapturedShadeAt()), so that the light changes smoothly across the mesh, and tinted by
// treeCapturedColor(). Pixels at the outline that show no triangle take a neighbour's triangle.
// image = lit(layer.rgb) + (1 - layer.a) * image.
//
// With `specular`, the mesh also reflects the sun as the training's specular reflection learned it
// (shaders/raster/specular.slang): the normal map tilts the smooth vertex normal where the ray meets the triangle, and
// the amplitude a, the brightest of the learned lights b and the learned exponent e (6 + exp(x)) give the highlight
// SUN_GLOSS a^2 b^2 max(dot(r, sun), 0)^e, scaled by the sun's light that reaches the point; SUN_GLOSS brings the
// highlight of the photographs' exposure to the scene's.

#define TREE_FN
#include "klartraum/tree/tree_shared.hpp"
#define TREE_LIGHT_SET 0
#define TREE_LIGHT_BINDING 7
#include "tree_light.slang"

struct Push {
    uint width;
    uint height;
    uint textureWidth;  // of the specular material
    uint textureHeight;
    uint specular;      // 1: add the specular reflection
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

[[vk::binding(0, 0)]] [[vk::image_format("rgba8")]] RWTexture2D<vec4> image;
[[vk::binding(1, 0)]] StructuredBuffer<vec4> layer;      // per pixel, row-major
[[vk::binding(2, 0)]] StructuredBuffer<uint> visibility; // per pixel: triangle + 1, 0 for none
[[vk::binding(3, 0)]] StructuredBuffer<vec4> positions;  // per vertex, in the morph's frame
[[vk::binding(4, 0)]] StructuredBuffer<uint> triangles;  // three vertices each
[[vk::binding(5, 0)]] StructuredBuffer<mat4> drawing; // world from the morph's frame, the morph's frame from clip
[[vk::binding(6, 0)]] StructuredBuffer<TreeRenderSettings> settings;
[[vk::binding(8, 0)]] StructuredBuffer<vec2> cornerUvs;     // three per triangle
[[vk::binding(9, 0)]] StructuredBuffer<vec4> material;      // per texel: normal offset xyz, amplitude
[[vk::binding(10, 0)]] StructuredBuffer<vec4> lights;       // four (direction, amplitude), then (log exponent)
[[vk::binding(11, 0)]] StructuredBuffer<vec4> vertexNormals; // per vertex, xyz: sums of area normals

// The scene's highlight relative to the photographs' (see above).
static const float SUN_GLOSS = 0.35f;

// The material at texture coordinate uv: bilinear, u wrapping and v clamped (as differentiable_shading.slang).
vec4 materialAt(vec2 uv) {
    float x = uv.x * float(pc.textureWidth) - 0.5f, y = uv.y * float(pc.textureHeight) - 0.5f;
    float fx = floor(x), fy = floor(y);
    int w = int(pc.textureWidth), h = int(pc.textureHeight);
    int i0 = ((int(fx) % w) + w) % w, i1 = (i0 + 1) % w;
    int j0 = clamp(int(fy), 0, h - 1), j1 = clamp(int(fy) + 1, 0, h - 1);
    vec2 f = vec2(x - fx, y - fy);
    vec4 top = mix(material[j0 * w + i0], material[j0 * w + i1], f.x);
    vec4 bottom = mix(material[j1 * w + i0], material[j1 * w + i1], f.x);
    return mix(top, bottom, f.y);
}

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
            if (pc.specular != 0u) {
                // Barycentric coordinates of the hit in the triangle.
                vec3 e0 = b - a, e1 = c - a, e2 = hit - a;
                float d00 = dot(e0, e0), d01 = dot(e0, e1), d11 = dot(e1, e1);
                float d20 = dot(e2, e0), d21 = dot(e2, e1);
                float denominator = max(d00 * d11 - d01 * d01, 1e-20f);
                float w1 = (d11 * d20 - d01 * d21) / denominator, w2 = (d00 * d21 - d01 * d20) / denominator;
                vec3 bary = clamp(vec3(1.0f - w1 - w2, w1, w2), 0.0f, 1.0f);
                bary /= bary.x + bary.y + bary.z;
                vec2 uv = bary.x * cornerUvs[3u * t] + bary.y * cornerUvs[3u * t + 1u] + bary.z * cornerUvs[3u * t + 2u];
                vec4 m4 = materialAt(uv);
                vec3 smooth = bary.x * vertexNormals[triangles[3u * t]].xyz +
                              bary.y * vertexNormals[triangles[3u * t + 1u]].xyz +
                              bary.z * vertexNormals[triangles[3u * t + 2u]].xyz;
                vec3 n = normalize((m * vec4(normalize(normalize(smooth) + m4.xyz), 0.0f)).xyz);
                vec3 toEye = -normalize((m * vec4(along, 0.0f)).xyz);
                vec3 r = 2.0f * dot(n, toEye) * n - toEye;
                float brightest = 0.0f;
                for (int k = 0; k < 4; ++k) {
                    brightest = max(brightest, lights[k].w * lights[k].w);
                }
                float sunLit = clamp(treeLightAll(s, world + normalize(s.sun.xyz) * (1.5f * s.lightGridMin.w)).x, 0.0f,
                                     1.0f);
                float highlight = SUN_GLOSS * m4.w * m4.w * brightest *
                                  pow(max(dot(r, normalize(s.sun.xyz)), 0.0f), 6.0f + exp(lights[4].x)) * sunLit;
                colour += highlight * min(s.sun.w, 1.0f) * vec3(s.lightColor.x, s.lightColor.y, s.lightColor.z);
            }
        }
    }
    vec4 below = image[p];
    image[p] = vec4(clamp(colour * top.a + (1.0f - top.a) * below.rgb, 0.0f, 1.0f), 1.0f);
}
