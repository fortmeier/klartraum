// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

struct PushConstants {
    uint width;
    uint height;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> inputData;
[[vk::binding(1, 0)]] [[vk::image_format("rgba8")]] WTexture2D<float4> outputImage;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    uint2 pixel = globalId.xy;
    if (pixel.x >= pc.width || pixel.y >= pc.height) {
        return;
    }

    uint planeSize = pc.width * pc.height;
    uint offset = pixel.y * pc.width + pixel.x;
    float3 color = float3(inputData[offset], inputData[planeSize + offset], inputData[2 * planeSize + offset]);
    outputImage.Store(int2(pixel), float4(clamp(color, 0.0, 1.0), 1.0));
}
