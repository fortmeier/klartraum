// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import glsl;

// GBufferMeshRenderer: the lit albedo, or a G-buffer in the encodings of the tree renderers (treeClassColor(),
// treeEncode16(), the depth of treeGBufferColor() and the coordinates of treeCoordinateColor()), or the world-space
// normal.

#define TREE_CAMERA_SET 0
#define TREE_CAMERA_BINDING 0
#include "../tree/tree_render_common.slang"

struct GBufferMeshSettings {
    vec4 output;    // x the output (GBufferOutput), y, z the depth window
    vec4 light;     // xyz towards the light, w the ambient share
    vec4 lights[2]; // the shadow-casting lights: xyz towards each, w its angular radius
    vec4 lighting;  // x, y their intensities, z their ambient share, w the triangles
};
[[vk::binding(1, 1)]] StructuredBuffer<GBufferMeshSettings> settings;

struct GBufferVertex {
    vec4 pos;
    vec4 normal;
    vec4 albedo;
    vec4 uv;
};
[[vk::binding(0, 1)]] StructuredBuffer<GBufferVertex> vertices;

// Whether the ray from `origin` along `direction` hits one of the scene's triangles (Moeller-Trumbore).
bool occluded(vec3 origin, vec3 direction, uint triangles) {
    for (uint t = 0u; t < triangles; ++t) {
        const vec3 a = vertices[3u * t].pos.xyz, b = vertices[3u * t + 1u].pos.xyz, c = vertices[3u * t + 2u].pos.xyz;
        const vec3 e1 = b - a, e2 = c - a, p = cross(direction, e2);
        const float det = dot(e1, p);
        if (abs(det) < 1e-9f)
            continue;
        const vec3 s = origin - a;
        const float u = dot(s, p) / det;
        if (u < 0.0f || u > 1.0f)
            continue;
        const vec3 q = cross(s, e1);
        const float v = dot(direction, q) / det;
        if (v < 0.0f || u + v > 1.0f)
            continue;
        if (dot(e2, q) / det > 1e-4f)
            return true;
    }
    return false;
}

// The share of the light's disc seen from the surface point: one ray for a point-like light, else 16 rays over the
// disc (a fixed Fibonacci pattern, so that the shadows do not flicker between frames).
float visibility(vec3 position, vec3 normal, vec4 light, uint triangles) {
    const vec3 towards = normalize(light.xyz);
    const vec3 origin = position + 1e-3f * normal;
    if (light.w <= 0.0f)
        return occluded(origin, towards, triangles) ? 0.0f : 1.0f;
    const vec3 helper = abs(towards.y) < 0.9f ? vec3(0.0f, 1.0f, 0.0f) : vec3(1.0f, 0.0f, 0.0f);
    const vec3 side = normalize(cross(towards, helper)), up = cross(side, towards);
    const float radius = tan(light.w);
    float seen = 0.0f;
    for (uint k = 0u; k < 16u; ++k) {
        const float r = radius * sqrt((float(k) + 0.5f) / 16.0f), angle = 2.39996323f * float(k);
        const vec3 direction = normalize(towards + r * (cos(angle) * side + sin(angle) * up));
        seen += occluded(origin, direction, triangles) ? 0.0f : 1.0f;
    }
    return seen / 16.0f;
}

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec4 inNormal;
layout(location = 2) in vec4 inAlbedo;
layout(location = 3) in vec4 inUv;

layout(location = 0) out vec4 outColor;

[shader("fragment")]
void main() {
    GBufferMeshSettings s = settings[0];
    uint output = uint(s.output.x + 0.5f);
    uint surfaceClass = uint(inAlbedo.w + 0.5f);
    if (output == 3u) {
        outColor = vec4(treeClassColor(surfaceClass), 1.0f);
        return;
    }
    if (output == 4u || output == 8u) {
        float d = clamp((length(inWorldPos - camera.cameraWorldPos.xyz) - s.output.y) /
                            max(s.output.z - s.output.y, 1e-6f), 0.0f, 1.0f);
        if (output == 8u) {
            outColor = vec4(treeEncode16(d), 0.0f, 1.0f);
        } else {
            outColor = vec4(d, d + 1.0f / 765.0f, d + 2.0f / 765.0f, 1.0f);
        }
        return;
    }
    if (output == 9u) {
        outColor = vec4(0.5f * normalize(inNormal.xyz) + 0.5f, 1.0f);
        return;
    }
    if (output == 10u || output == 11u) {
        const vec3 n = normalize(inNormal.xyz);
        const uint triangles = uint(s.lighting.w + 0.5f);
        const float v1 = visibility(inWorldPos, n, s.lights[0], triangles);
        const float v2 = visibility(inWorldPos, n, s.lights[1], triangles);
        if (output == 10u) {
            outColor = vec4(v1, v2, 0.0f, 1.0f);
            return;
        }
        const float lit = s.lighting.z + s.lighting.x * v1 * max(dot(n, normalize(s.lights[0].xyz)), 0.0f) +
                          s.lighting.y * v2 * max(dot(n, normalize(s.lights[1].xyz)), 0.0f);
        outColor = vec4(inAlbedo.xyz * lit, 1.0f);
        return;
    }
    if (output == 6u || output == 7u) {
        // Repeating coordinates wrap; the others stay in [0, 1] (a face's far edge keeps 1 instead of wrapping to 0).
        vec2 uv = inUv.z > 0.5f ? fract(inUv.xy) : clamp(inUv.xy, 0.0f, 1.0f);
        uint id = uint(inNormal.w + 0.5f);
        bool second = output == 7u;
        vec2 bytes = treeEncode16(second ? uv.y : uv.x);
        float idByte = float(second ? (id & 255u) : ((id >> 8u) & 255u)) / 255.0f;
        outColor = vec4(bytes.x, bytes.y, idByte, 1.0f);
        return;
    }
    vec3 normal = normalize(inNormal.xyz);
    float lit = s.light.w + (1.0f - s.light.w) * max(dot(normal, s.light.xyz), 0.0f);
    outColor = vec4(inAlbedo.xyz * lit, 1.0f);
}
