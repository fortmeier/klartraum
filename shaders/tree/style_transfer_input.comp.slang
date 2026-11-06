// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import glsl;

// The input of StyleTransfer's network at the frame's size or half of it (scale 1 or 2), NCHW: the colour averaged
// over scale x scale pixels and, for 11 channels, the classes of the G-buffer one-hot (the class whose colour is
// nearest to pixel (scale x + scale / 2, scale y + scale / 2)) and the depth averaged over scale x scale pixels.

#define TREE_FN
#include "klartraum/tree/tree_shared.hpp"

struct Push {
    uint frameWidth;
    uint frameHeight;
    uint width;
    uint height;
    uint channels;
    uint scale;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

[[vk::binding(0, 0)]] [[vk::image_format("rgba8")]] RWTexture2D<float4> colorImage;
[[vk::binding(1, 0)]] [[vk::image_format("rgba8")]] RWTexture2D<float4> classImage;
[[vk::binding(2, 0)]] [[vk::image_format("rgba8")]] RWTexture2D<float4> depthImage;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> outputData;

vec3 average(RWTexture2D<float4> image, ivec2 at) {
    if (pc.scale == 1u) {
        return image[at].rgb;
    }
    return 0.25f * (image[at].rgb + image[at + ivec2(1, 0)].rgb + image[at + ivec2(0, 1)].rgb +
                    image[at + ivec2(1, 1)].rgb);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main() {
    uvec2 pixel = gl_GlobalInvocationID.xy;
    if (pixel.x >= pc.width || pixel.y >= pc.height) {
        return;
    }
    ivec2 at = ivec2(pixel) * int(pc.scale);
    uint plane = pc.width * pc.height;
    uint offset = pixel.y * pc.width + pixel.x;
    vec3 color = average(colorImage, at);
    outputData[offset] = color.r;
    outputData[plane + offset] = color.g;
    outputData[2u * plane + offset] = color.b;
    if (pc.channels == 11u) {
        vec3 c = classImage[at + ivec2(int(pc.scale / 2u))].rgb;
        uint nearest = 0u;
        float best = 1e9f;
        for (uint k = 0u; k < TREE_CLASS_COUNT; ++k) {
            vec3 d = treeClassColor(k) - c;
            if (dot(d, d) < best) {
                best = dot(d, d);
                nearest = k;
            }
        }
        for (uint k = 0u; k < TREE_CLASS_COUNT; ++k) {
            outputData[(3u + k) * plane + offset] = k == nearest ? 1.0f : 0.0f;
        }
        outputData[(3u + TREE_CLASS_COUNT) * plane + offset] = average(depthImage, at).r;
    }
}
