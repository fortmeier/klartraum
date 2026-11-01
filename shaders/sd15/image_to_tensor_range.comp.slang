// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// RGB of an image as a [3, height, width] tensor with values from `low` to
// `high` (bgraToTensor() on the GPU).

struct PushConstants {
    uint width;
    uint height;
    float low;
    float high;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] [[vk::image_format("rgba8")]] RWTexture2D<float4> inputImage;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> outputData;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    uint2 pixel = globalId.xy;
    if (pixel.x >= pc.width || pixel.y >= pc.height) {
        return;
    }
    float3 color = inputImage.Load(int2(pixel)).rgb;
    uint plane = pc.width * pc.height;
    uint offset = pixel.y * pc.width + pixel.x;
    outputData[offset] = pc.low + (pc.high - pc.low) * color.r;
    outputData[plane + offset] = pc.low + (pc.high - pc.low) * color.g;
    outputData[2 * plane + offset] = pc.low + (pc.high - pc.low) * color.b;
}
