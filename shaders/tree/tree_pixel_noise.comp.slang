// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import glsl;
import scalar_buffer;

// Noise at the scale of a pixel that stays with the world, for the enhancing network's noise input: for every pixel
// with a surface in the splatting's depth buffer (NDC depth, +inf where there is none), its world position is
// reconstructed and value noise of it is written into feature B's third channel, with cells about one pixel wide.
// As the pixel's footprint changes with distance, the two octaves around it are blended by the fractional octave,
// so the noise does not jump. The coordinates are rotated so that the lattice does not line up with the image's
// rows. Pixels without a surface keep the splatted noise of tree_gaussian_features.comp.

#define TREE_CAMERA_SET 0
#define TREE_CAMERA_BINDING 1
#include "tree_render_common.slang"

[[vk::binding(0, 0)]] ScalarBuffer<float> depth;
[[vk::binding(2, 0)]] [[vk::image_format("rgba8")]] RWTexture2D<float4> featureB;

struct Push {
    uint width;
    uint height;
    float cellUnits; // the finest noise cell, in scene units
    uint pad;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

uint pcg(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float latticeValue(ivec3 c) { return float(pcg(uint(c.x) + pcg(uint(c.y) + pcg(uint(c.z))))) / 4294967296.0f; }

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

// A fixed rotation (about 37 degrees about (1, 2, 3)), so that the lattice is not aligned with the world's axes.
static const mat3 ROTATION = mat3(0.8235f, 0.4718f, -0.3149f, -0.3963f, 0.8708f, 0.2908f, 0.4114f, -0.1384f, 0.9009f);

[shader("compute")]
[numthreads(8, 8, 1)]
void main() {
    uvec2 pixel = gl_GlobalInvocationID.xy;
    if (pixel.x >= pc.width || pixel.y >= pc.height) {
        return;
    }
    float ndc = depth[pixel.y * pc.width + pixel.x];
    if (isinf(ndc) || isnan(ndc)) {
        return;
    }
    vec2 xy = (vec2(pixel) + 0.5f) / vec2(pc.width, pc.height) * 2.0f - 1.0f;
    vec4 world = inverse(camera.proj * camera.view) * vec4(xy, ndc, 1.0f);
    vec3 p = ROTATION * (world.xyz / world.w);
    float viewDistance = camera.proj[3][2] / (ndc + camera.proj[2][2]);
    float footprint = 2.0f * viewDistance / (abs(camera.proj[1][1]) * float(pc.height));
    float octave = max(log2(footprint / pc.cellUnits), 0.0f);
    float level = floor(octave);
    float fine = valueNoise(p / (pc.cellUnits * exp2(level)) + vec3(level * 17.31f));
    float coarse = valueNoise(p / (pc.cellUnits * exp2(level + 1.0f)) + vec3((level + 1.0f) * 17.31f));
    vec4 value = featureB[ivec2(pixel)];
    value.b = mix(fine, coarse, octave - level);
    featureB[ivec2(pixel)] = value;
}
