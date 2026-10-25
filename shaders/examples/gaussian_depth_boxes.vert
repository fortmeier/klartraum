// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#version 450

layout(push_constant) uniform Box { mat4 mvp; vec4 color; } box;
layout(location = 0) out vec3 color;

void main() {
    const vec3 corners[8] = vec3[](
        vec3(-1,-1,-1), vec3(1,-1,-1), vec3(1,1,-1), vec3(-1,1,-1),
        vec3(-1,-1, 1), vec3(1,-1, 1), vec3(1,1, 1), vec3(-1,1, 1));
    const int indices[36] = int[](
        4,5,6, 4,6,7, 1,0,3, 1,3,2, 0,4,7, 0,7,3,
        5,1,2, 5,2,6, 7,6,2, 7,2,3, 0,1,5, 0,5,4);
    const float shade[6] = float[](1.0, 0.55, 0.72, 0.85, 1.12, 0.45);
    gl_Position = box.mvp * vec4(corners[indices[gl_VertexIndex]], 1);
    color = box.color.rgb * shade[gl_VertexIndex / 6];
}
