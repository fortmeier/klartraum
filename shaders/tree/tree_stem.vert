// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#version 450

// A stem segment as a cubic Hermite spline tube (Kuth et al., Sec. 3.2): a
// grid of around x along quads over the arc of the tube that faces the camera.
// The rows at the two rings use only the ring's data, so a ring's vertices are
// the same in both segments that share it; the tessellation factors are
// fractional with odd spacing, so vertices fade in smoothly.

#define TREE_CAMERA_SET 0
#define TREE_CAMERA_BINDING 0
#include "tree_render_common.glsl"

layout(std430, set = 1, binding = 0) readonly buffer Segments {
    SegmentRecord segments[];
};
layout(std430, set = 1, binding = 1) readonly buffer Lists {
    uint lists[];
};
layout(std430, set = 1, binding = 2) readonly buffer Params {
    TreeTypeParams params[];
};
layout(std430, set = 1, binding = 3) readonly buffer Settings {
    TreeRenderSettings settings[];
};

layout(push_constant) uniform Push {
    uint bucket;
    uint capacity;
}
pc;

layout(location = 0) out vec3 outWorldPos;
layout(location = 1) out vec3 outNormal;
layout(location = 2) out vec3 outColor;
layout(location = 3) out vec3 outBark; // x angle around, y along the stem in world units, z radius
layout(location = 4) out vec3 outTangent; // around the stem
layout(location = 5) flat out uint outType;

void main() {
    uint around = treeStemBucketAround(pc.bucket);
    uint along = treeStemBucketAlong(pc.bucket);
    SegmentRecord d = segments[lists[pc.bucket * pc.capacity + gl_InstanceIndex]];
    TreeRenderSettings s = settings[0];
    StemRing a = stemFrom(d);
    StemRing b = stemTo(d);

    float f = clamp(treeRingFactor(a, s), 1.0f, float(around));
    float t = clamp(treeRingFactor(b, s), 1.0f, float(around));
    float u = 0.5f * (f + t);
    float v = clamp(treeSegmentAlongFactor(a, b, s), 1.0f, float(along));

    uint cell = uint(gl_VertexIndex) / 6u;
    uint corner = uint(gl_VertexIndex) % 6u;
    uint column = cell % around + ((corner == 1u || corner == 2u || corner == 4u) ? 1u : 0u);
    uint row = cell / around + ((corner == 2u || corner == 4u || corner == 5u) ? 1u : 0u);

    float segmentLength = length(b.pos - a.pos);
    StemRing ring;
    float factor;
    float w;
    if (row == 0u) {
        ring = a;
        factor = f;
        w = 0.0f;
    } else if (row >= along || treeTessSpacing(row, v, along) >= 1.0f) {
        ring = b;
        factor = t;
        w = 1.0f;
    } else {
        w = treeTessSpacing(row, v, along);
        ring.pos = treeHermite(a.pos, b.pos, qforward(a.rot) * segmentLength, qforward(b.rot) * segmentLength, w);
        ring.rot = qnlerp(a.rot, b.rot, w);
        ring.radius = mix(a.radius, b.radius, w);
        ring.drdz = mix(a.drdz, b.drdz, w);
        factor = u;
    }

    float theta = around == 1u ? 0.5f * TREE_PI : treeRingTheta(ring);
    vec3 x = qrot(ring.rot, vec3(1.0f, 0.0f, 0.0f));
    vec3 y = qrot(ring.rot, vec3(0.0f, 1.0f, 0.0f));
    vec3 toCamera = camera.cameraWorldPos.xyz - ring.pos;
    float phiCenter = atan(dot(toCamera, y), dot(toCamera, x));
    float phi = phiCenter - theta + 2.0f * theta * treeTessSpacing(column, factor, around);

    vec3 radial = cos(phi) * x + sin(phi) * y;
    // The ring's own z along the stem, so that a ring's lobes are the same in both its segments.
    float z = row == 0u ? d.shape.z : d.shape.z + w * segmentLength / d.shape.w;
    if (w >= 1.0f) {
        z = d.shape.z + segmentLength / d.shape.w;
    }
    TreeTypeParams p = params[treeType(d.ids)];
    float radius = ring.radius * treeTrunkLobes(p, treeLevel(d.ids), z, phi);
    vec3 worldPos = ring.pos + radius * radial;
    outWorldPos = worldPos;
    outNormal = normalize(radial - ring.drdz * qforward(ring.rot));
    outColor = p.barkColor.rgb;
    outBark = vec3(phi, d.shape.w * z, ring.radius);
    outTangent = -sin(phi) * x + cos(phi) * y;
    outType = treeType(d.ids);
    gl_Position = camera.proj * camera.view * vec4(worldPos, 1.0f);
}
