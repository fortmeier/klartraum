// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import glsl;

// The static scene meshes: a triangle list of TreeSceneVertex.

#define TREE_CAMERA_SET 0
#define TREE_CAMERA_BINDING 0
#include "tree_render_common.slang"

[[vk::binding(0, 1)]] StructuredBuffer<TreeSceneVertex> vertices;

layout(location = 0) out vec3 outWorldPos;
layout(location = 1) out vec3 outNormal;
layout(location = 2) out vec4 outColor;

[shader("vertex")]
void main() {
    TreeSceneVertex v = vertices[gl_VertexIndex];
    outWorldPos = v.pos.xyz;
    outNormal = v.normal.xyz;
    outColor = v.color;
    gl_Position = camera.proj * camera.view * vec4(v.pos.xyz, 1.0f);
}
