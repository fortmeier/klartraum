// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import glsl;

// The input of the neural enhancer of distant trees (scripts/neural/train_tree_enhancer.py): the Gaussian image's
// RGB and its feature images (BonsaiRenderer::getFeatureTarget(): coverage, tree, level of detail; Gaussian size,
// depth; and for networks of 9 input channels the splatted noise in feature B's third channel) as an NCHW tensor of
// pc.channels planes in [0, 1], in the order the network was trained with.

struct PushConstants {
    uint width;
    uint height;
    uint channels;
    uint pad;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] [[vk::image_format("rgba8")]] RWTexture2D<float4> colorImage;
[[vk::binding(1, 0)]] [[vk::image_format("rgba8")]] RWTexture2D<float4> featureA;
[[vk::binding(2, 0)]] [[vk::image_format("rgba8")]] RWTexture2D<float4> featureB;

[[vk::binding(3, 0)]] RWStructuredBuffer<float> outputData;

[shader("compute")]
[numthreads(8, 8, 1)]
void main() {
    uvec2 pixel = gl_GlobalInvocationID.xy;
    if (pixel.x >= pc.width || pixel.y >= pc.height) {
        return;
    }
    vec3 color = colorImage[ivec2(pixel)].rgb;
    vec3 a = featureA[ivec2(pixel)].rgb;
    vec3 b = featureB[ivec2(pixel)].rgb;
    uint plane = pc.width * pc.height;
    uint offset = pixel.y * pc.width + pixel.x;
    outputData[offset] = color.r;
    outputData[plane + offset] = color.g;
    outputData[2u * plane + offset] = color.b;
    outputData[3u * plane + offset] = a.r;
    outputData[4u * plane + offset] = a.g;
    outputData[5u * plane + offset] = a.b;
    outputData[6u * plane + offset] = b.r;
    outputData[7u * plane + offset] = b.g;
    if (pc.channels > 8u) {
        outputData[8u * plane + offset] = b.b;
    }
}
