// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#version 450
layout(push_constant) uniform Params { vec4 color; float depth; } pc;
layout(location = 0) out vec4 color;
void main() {
    const vec2 positions[3] = vec2[](vec2(-1, -1), vec2(3, -1), vec2(-1, 3));
    gl_Position = vec4(positions[gl_VertexIndex], pc.depth, 1);
    color = pc.color;
}
