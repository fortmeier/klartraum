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
    vec4 output; // x the output (GBufferOutput), y, z the depth window
    vec4 light;  // xyz towards the light, w the ambient share
};
[[vk::binding(1, 1)]] StructuredBuffer<GBufferMeshSettings> settings;

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
