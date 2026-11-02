// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import glsl;
import scalar_buffer;

// Per-pixel features of a Gaussian rendering, for a network that enhances it: every Gaussian gets two feature
// colours instead of its own, with its opacity, so that splatting them blends each feature as the colours are
// blended. A: coverage (1), tree (1 for the generated trees' and the library copies' Gaussians, else 0), level of
// detail (1 generated, 0.5 copy, 0 else). B: the Gaussian's size on screen, log2(1 + its largest sigma in pixels)
// / 8, its depth below the camera, log2(1 + view depth / pc.depthUnit) / 16, and noise: value noise in [0, 1] of the
// Gaussian's position with cells of half a pc.depthUnit, so that it stays with the Gaussian from frame to frame,
// whatever order the Gaussians are written in, and changes smoothly as wind moves it (the enhancing network draws
// detail from it that the rest of its input does not determine). The colours are written as degree 0
// spherical harmonics, as the splatting reads them.

#define TREE_CAMERA_SET 0
#define TREE_CAMERA_BINDING 3
#include "tree_render_common.slang"

[[vk::binding(0, 0)]] ScalarBuffer<vec3> positions;
[[vk::binding(1, 0)]] ScalarBuffer<vec3> scales;
[[vk::binding(2, 0)]] ScalarBuffer<vec4> colAlpha;
[[vk::binding(4, 0)]] RWScalarBuffer<vec4> featureA;
[[vk::binding(5, 0)]] RWScalarBuffer<vec4> featureB;

struct Push {
    uint count;
    uint treeBegin; // the generated trees' slots [treeBegin, treeEnd)
    uint treeEnd;
    uint copyBegin; // the library copies' slots [copyBegin, copyEnd)
    uint copyEnd;
    float halfHeight; // half the image's height in pixels
    float depthUnit; // units in which the depth is encoded (a metre)
    uint pad;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

static const float SH_C0 = 0.28209479f;

vec4 asColour(vec3 feature, float alpha) { return vec4((feature - vec3(0.5f)) / SH_C0, alpha); }

// PCG hash (Jarzynski and Olano 2020).
uint pcg(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// A value in [0, 1) per lattice cell.
float latticeValue(ivec3 c) { return float(pcg(uint(c.x) + pcg(uint(c.y) + pcg(uint(c.z))))) / 4294967296.0f; }

// Trilinear value noise with smoothstep weights.
float valueNoise(vec3 p) {
    ivec3 c = ivec3(floor(p));
    vec3 f = p - floor(p);
    vec3 w = f * f * (3.0f - 2.0f * f);
    float x00 = mix(latticeValue(c), latticeValue(c + ivec3(1, 0, 0)), w.x);
    float x10 = mix(latticeValue(c + ivec3(0, 1, 0)), latticeValue(c + ivec3(1, 1, 0)), w.x);
    float x01 = mix(latticeValue(c + ivec3(0, 0, 1)), latticeValue(c + ivec3(1, 0, 1)), w.x);
    float x11 = mix(latticeValue(c + ivec3(0, 1, 1)), latticeValue(c + ivec3(1, 1, 1)), w.x);
    return mix(mix(x00, x10, w.y), mix(x01, x11, w.y), w.z);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.count) {
        return;
    }
    float alpha = colAlpha[i].w;
    bool generated = i >= pc.treeBegin && i < pc.treeEnd;
    bool copy = i >= pc.copyBegin && i < pc.copyEnd;
    float lod = generated ? 1.0f : (copy ? 0.5f : 0.0f);
    featureA[i] = asColour(vec3(1.0f, generated || copy ? 1.0f : 0.0f, lod), alpha);
    vec3 s = scales[i];
    float depth = max(treeViewDepth(positions[i]), 1e-6f);
    float sigmaPixels = max(s.x, max(s.y, s.z)) * abs(camera.proj[1][1]) * pc.halfHeight / depth;
    featureB[i] = asColour(vec3(clamp(log2(1.0f + sigmaPixels) / 8.0f, 0.0f, 1.0f),
                                clamp(log2(1.0f + depth / pc.depthUnit) / 16.0f, 0.0f, 1.0f), valueNoise(positions[i] / (0.5f * pc.depthUnit))),
                           alpha);
}
