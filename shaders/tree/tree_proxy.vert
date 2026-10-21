// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#version 450

// A proxy tree per visible site (TreeProxies): 72 vertices, a trunk of six
// sides and a crown of two six-sided cones meeting at its widest ring
// (treeProxyShape()), the ring's radius varied per side; flat normals per
// triangle.

#define TREE_CAMERA_SET 0
#define TREE_CAMERA_BINDING 0
#include "tree_render_common.glsl"

layout(std430, set = 1, binding = 0) readonly buffer Instances {
    TreeSite instances[];
};
layout(std430, set = 1, binding = 1) readonly buffer Settings {
    TreeRenderSettings settings[];
};

layout(location = 0) out vec3 outWorldPos;
layout(location = 1) out vec3 outNormal;
layout(location = 2) out vec3 outAlbedo;
layout(location = 3) out float outCoverage; // of the crown: bare broadleaves are sparse

vec3 ring(float angle, float radius, float y) {
    return vec3(radius * cos(angle), y, radius * sin(angle));
}

void main() {
    TreeSite site = instances[gl_InstanceIndex];
    float h = site.posHeight.w;
    vec4 shape = treeProxyShape(site.kind.x);
    int v = gl_VertexIndex;
    bool trunk = v < 36;
    int local = v % 36;
    int tri = local / 3;
    int corner = local % 3;
    int side = tri / 2;
    bool second = (tri % 2) == 1;
    float a0 = site.kind.y + float(side) * (TREE_PI / 3.0f);
    float a1 = a0 + TREE_PI / 3.0f;
    uint key = floatBitsToUint(site.kind.y);
    vec3 p[3];
    if (trunk) {
        float rt = shape.w * h;
        float top = (shape.x + 0.05f) * h;
        if (!second) {
            p[0] = ring(a0, rt, 0.0f);
            p[1] = ring(a1, rt, 0.0f);
            p[2] = ring(a1, rt, top);
        } else {
            p[0] = ring(a0, rt, 0.0f);
            p[1] = ring(a1, rt, top);
            p[2] = ring(a0, rt, top);
        }
    } else {
        float r0 = shape.z * h * (0.8f + 0.4f * treeRand(key, uint(side)));
        float r1 = shape.z * h * (0.8f + 0.4f * treeRand(key, uint((side + 1) % 6)));
        vec3 q0 = ring(a0, r0, shape.y * h);
        vec3 q1 = ring(a1, r1, shape.y * h);
        if (!second) {
            p[0] = vec3(0.0f, shape.x * h, 0.0f);
            p[1] = q1;
            p[2] = q0;
        } else {
            p[0] = q0;
            p[1] = q1;
            p[2] = vec3(0.0f, h, 0.0f);
        }
    }
    vec3 normal = normalize(cross(p[1] - p[0], p[2] - p[0]));
    vec3 middle = (p[0] + p[1] + p[2]) / 3.0f;
    if (dot(normal, vec3(middle.x, 0.0f, middle.z)) < 0.0f) {
        normal = -normal;
    }
    vec3 world = site.posHeight.xyz + p[corner];
    vec4 crown = treeProxyCrown(site, settings[0].season.x);
    // Lower in the crown is darker: less light reaches there.
    float shade = 0.6f + 0.4f * clamp(p[corner].y / h, 0.0f, 1.0f);
    outAlbedo = trunk ? vec3(0.09f, 0.07f, 0.05f) : vec3(crown) * shade;
    // Fading in where the canopy fades out (kind.w).
    outCoverage = (trunk ? 1.0f : crown.w) * site.kind.w;
    outWorldPos = world;
    outNormal = normal;
    gl_Position = camera.proj * camera.view * vec4(world, 1.0f);
}
