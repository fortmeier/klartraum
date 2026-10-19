// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Shared by the tree render passes: the camera and the per-ring stem
// tessellation, which the classification pass and the vertex shader must
// compute identically so that neighbouring segments meet without cracks.
// The includer defines TREE_CAMERA_SET and TREE_CAMERA_BINDING.

#extension GL_GOOGLE_include_directive : require

#define TREE_FN
#include "klartraum/tree/tree_shared.hpp"

layout(std140, set = TREE_CAMERA_SET, binding = TREE_CAMERA_BINDING) uniform TreeCamera {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec4 cameraWorldPos;
}
camera;

struct StemRing {
    vec3 pos;
    vec4 rot;
    float radius;
    float drdz;
};

StemRing stemFrom(SegmentRecord d) {
    StemRing r;
    r.pos = d.fromPosRadius.xyz;
    r.rot = d.fromRot;
    r.radius = d.fromPosRadius.w;
    r.drdz = d.shape.x;
    return r;
}

StemRing stemTo(SegmentRecord d) {
    StemRing r;
    r.pos = d.toPosRadius.xyz;
    r.rot = d.toRot;
    r.radius = d.toPosRadius.w;
    r.drdz = d.shape.y;
    return r;
}

float treeViewDepth(vec3 p) {
    return -(camera.view * vec4(p, 1.0f)).z;
}

// Pixels per world unit at view depth 1.
float treeFocalPixels(TreeRenderSettings settings) {
    return abs(camera.proj[1][1]) * 0.5f * settings.viewport.y;
}

float treeRingPixelRadius(StemRing ring, TreeRenderSettings settings) {
    return ring.radius * treeFocalPixels(settings) / max(treeViewDepth(ring.pos), 1e-4f);
}

float treeRingTheta(StemRing ring) {
    vec3 toCamera = normalize(camera.cameraWorldPos.xyz - ring.pos);
    return treeOpeningAngle(ring.drdz, dot(qforward(ring.rot), toCamera));
}

// Fractional number of quads around the visible arc of a ring.
float treeRingFactor(StemRing ring, TreeRenderSettings settings) {
    return 2.0f * treeRingTheta(ring) * treeRingPixelRadius(ring, settings) / settings.viewport.z;
}

// Fractional number of quads along a segment: its length in pixels.
float treeSegmentAlongFactor(StemRing a, StemRing b, TreeRenderSettings settings) {
    vec4 ca = camera.proj * camera.view * vec4(a.pos, 1.0f);
    vec4 cb = camera.proj * camera.view * vec4(b.pos, 1.0f);
    float pixels;
    if (ca.w > 1e-4f && cb.w > 1e-4f) {
        vec2 pa = ca.xy / ca.w * 0.5f * settings.viewport.xy;
        vec2 pb = cb.xy / cb.w * 0.5f * settings.viewport.xy;
        pixels = length(pa - pb);
    } else {
        pixels = length(a.pos - b.pos) * treeFocalPixels(settings) / 1e-2f;
    }
    return pixels / settings.viewport.z;
}
