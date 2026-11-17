// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import glsl;

// GBufferMeshRenderer: a static triangle list of GBufferVertex (gbuffer_mesh_renderer.hpp).

#define TREE_CAMERA_SET 0
#define TREE_CAMERA_BINDING 0
#include "../tree/tree_render_common.slang"

struct GBufferVertex {
    vec4 pos;    // xyz, w the surface class
    vec4 normal; // xyz, w the object id
    vec4 albedo; // rgb
    vec4 uv;     // xy, z 1 where the coordinates repeat
};
[[vk::binding(0, 1)]] StructuredBuffer<GBufferVertex> vertices;

layout(location = 0) out vec3 outWorldPos;
layout(location = 1) out vec4 outNormal; // xyz, w the object id
layout(location = 2) out vec4 outAlbedo; // rgb, w the surface class
layout(location = 3) out vec4 outUv;

[shader("vertex")]
void main() {
    GBufferVertex v = vertices[gl_VertexIndex];
    outWorldPos = v.pos.xyz;
    outNormal = v.normal;
    outAlbedo = vec4(v.albedo.xyz, v.pos.w);
    outUv = v.uv;
    gl_Position = camera.proj * camera.view * vec4(v.pos.xyz, 1.0f);
}
