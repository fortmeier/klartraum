// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import glsl;

// The input of FieldStyle's network (scripts/neural/train_watercolor_field.py: pack()), 41 planes at the frame's
// size, NCHW: the colour (3), the classes of the G-buffer one-hot (7, the class whose colour is nearest), the depth
// (1, the distance from the camera mapped from [2, 16] to [0, 1], 1 for background), the features of the two world
// grids at the surface point the pixel shows (16, 0 for background), the paper's features at the pixel (4, 0 on
// surfaces; a tile repeated across the frame, two pixels per texel), the frame's conditions (7) and the logit of the colour (3, clamped to [0.02, 0.98]).
//
// A pixel shows a surface where its class is not background and the surfaces cover more than 5 % of it (the coverage
// G-buffer). Its distance comes from the depth G-buffer over black between the frame's near and far: the mean of its
// three channels (tree::treeGBufferColor()) divided by the coverage, less the channels' mean offset of one step. The
// surface point lies at that distance along the pixel centre's ray.

#define TREE_FN
#include "klartraum/tree/tree_shared.hpp"

struct Push {
    uint width;
    uint height;
    uint coarseX, coarseY, coarseZ;
    uint fineX, fineY, fineZ;
    uint paperWidth, paperHeight;
    uint fineOffset;  // first float of the fine grid in `field`
    uint paperOffset; // first float of the paper in `field`
    float lowX, lowY, lowZ;
    float highX, highY, highZ;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

[[vk::binding(0, 0)]] [[vk::image_format("rgba8")]] RWTexture2D<float4> colorImage;
[[vk::binding(1, 0)]] [[vk::image_format("rgba8")]] RWTexture2D<float4> classImage;
[[vk::binding(2, 0)]] [[vk::image_format("rgba8")]] RWTexture2D<float4> depthImage;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> outputData;
// The frame: inverse of the camera's clip from world (columns 0-3), the eye (4), near and far of the depth G-buffer
// (5.xy), the conditions (6.xyzw, 7.xyz).
[[vk::binding(4, 0)]] StructuredBuffer<vec4> frame;
// The coarse grid, the fine grid (eight features per corner, x fastest) and the paper (four planes).
[[vk::binding(5, 0)]] StructuredBuffer<float> field;
[[vk::binding(6, 0)]] [[vk::image_format("rgba8")]] RWTexture2D<float4> coverageImage;

static const uint CHANNELS = 8u;

// The trilinear look-up of a grid of size (sx, sy, sz) over the box at `p`, as the training's Grid.
void lookUp(uint offset, uvec3 size, vec3 p, inout float features[16], uint first) {
    vec3 low = vec3(pc.lowX, pc.lowY, pc.lowZ), high = vec3(pc.highX, pc.highY, pc.highZ);
    vec3 scale = (vec3(size) - 1.0f) / (high - low);
    vec3 g = min(max((p - low) * scale, vec3(0.0f)), vec3(size) - 1.001f);
    vec3 base = floor(g);
    vec3 f = g - base;
    uvec3 i = uvec3(base);
    for (uint c = 0u; c < CHANNELS; ++c) {
        features[first + c] = 0.0f;
    }
    for (uint dz = 0u; dz < 2u; ++dz) {
        for (uint dy = 0u; dy < 2u; ++dy) {
            for (uint dx = 0u; dx < 2u; ++dx) {
                float w = (dx == 1u ? f.x : 1.0f - f.x) * (dy == 1u ? f.y : 1.0f - f.y) *
                          (dz == 1u ? f.z : 1.0f - f.z);
                uint index = ((i.z + dz) * size.y + (i.y + dy)) * size.x + (i.x + dx);
                for (uint c = 0u; c < CHANNELS; ++c) {
                    features[first + c] += w * field[offset + index * CHANNELS + c];
                }
            }
        }
    }
}

// The paper's plane `c` at texel (x, y) of its tile, repeated across the frame.
float paperAt(uint c, int x, int y) {
    int w = int(pc.paperWidth), h = int(pc.paperHeight);
    uint i = uint(((x % w) + w) % w), j = uint(((y % h) + h) % h);
    return field[pc.paperOffset + (c * pc.paperHeight + j) * pc.paperWidth + i];
}

// Frame pixels per paper texel along each side.
static const float PAPER_PIXELS = 2.0f;

[shader("compute")]
[numthreads(8, 8, 1)]
void main() {
    uvec2 pixel = gl_GlobalInvocationID.xy;
    if (pixel.x >= pc.width || pixel.y >= pc.height) {
        return;
    }
    uint plane = pc.width * pc.height;
    uint offset = pixel.y * pc.width + pixel.x;
    ivec2 at = ivec2(pixel);
    vec3 color = colorImage[at].rgb;

    vec3 c = classImage[at].rgb;
    uint nearest = 0u;
    float best = 1e9f;
    for (uint k = 0u; k < TREE_CLASS_COUNT; ++k) {
        vec3 d = treeClassColor(k) - c;
        if (dot(d, d) < best) {
            best = dot(d, d);
            nearest = k;
        }
    }
    float coverage = coverageImage[at].r;
    bool surface = nearest > 0u && coverage > 0.05f;
    vec3 depth = depthImage[at].rgb;
    float grey = clamp((depth.r + depth.g + depth.b) / 3.0f / max(coverage, 1e-6f) - 1.0f / 255.0f, 0.0f, 1.0f);
    float near = frame[5].x, far = frame[5].y;
    float distance = near + grey * (far - near);

    float features[16];
    for (uint k = 0u; k < 16u; ++k) {
        features[k] = 0.0f;
    }
    float paper[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (surface) {
        mat4 inverseCamera = mat4(frame[0], frame[1], frame[2], frame[3]);
        vec2 ndc = (vec2(pixel) + 0.5f) / vec2(float(pc.width), float(pc.height)) * 2.0f - 1.0f;
        vec4 farPoint = inverseCamera * vec4(ndc, 1.0f, 1.0f);
        vec3 eye = frame[4].xyz;
        vec3 direction = normalize(farPoint.xyz / farPoint.w - eye);
        vec3 p = eye + direction * distance;
        lookUp(0u, uvec3(pc.coarseX, pc.coarseY, pc.coarseZ), p, features, 0u);
        lookUp(pc.fineOffset, uvec3(pc.fineX, pc.fineY, pc.fineZ), p, features, CHANNELS);
    } else {
        float sx = (float(pixel.x) + 0.5f) / PAPER_PIXELS - 0.5f;
        float sy = (float(pixel.y) + 0.5f) / PAPER_PIXELS - 0.5f;
        int x0 = int(floor(sx)), y0 = int(floor(sy));
        float fx = sx - float(x0), fy = sy - float(y0);
        for (uint k = 0u; k < 4u; ++k) {
            float top = mix(paperAt(k, x0, y0), paperAt(k, x0 + 1, y0), fx);
            float bottom = mix(paperAt(k, x0, y0 + 1), paperAt(k, x0 + 1, y0 + 1), fx);
            paper[k] = mix(top, bottom, fy);
        }
    }

    float values[41];
    for (uint k = 0u; k < 3u; ++k) {
        values[k] = color[k];
        float clamped = clamp(color[k], 0.02f, 0.98f);
        values[38u + k] = log(clamped / (1.0f - clamped));
    }
    for (uint k = 0u; k < 7u; ++k) {
        values[3u + k] = k == nearest ? 1.0f : 0.0f;
    }
    values[10] = surface ? clamp((distance - 2.0f) / 14.0f, 0.0f, 1.0f) : 1.0f;
    for (uint k = 0u; k < 16u; ++k) {
        values[11u + k] = features[k];
    }
    for (uint k = 0u; k < 4u; ++k) {
        values[27u + k] = paper[k];
    }
    vec4 first = frame[6], second = frame[7];
    values[31] = first.x;
    values[32] = first.y;
    values[33] = first.z;
    values[34] = first.w;
    values[35] = second.x;
    values[36] = second.y;
    values[37] = second.z;
    for (uint k = 0u; k < 41u; ++k) {
        outputData[k * plane + offset] = values[k];
    }
}
