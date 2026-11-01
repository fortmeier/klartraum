// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

struct PushConstants {
    uint width;
    uint height;
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
    uint planeSize = pc.width * pc.height;
    uint offset = pixel.y * pc.width + pixel.x;
    outputData[offset] = color.r;
    outputData[planeSize + offset] = color.g;
    outputData[2 * planeSize + offset] = color.b;
}
