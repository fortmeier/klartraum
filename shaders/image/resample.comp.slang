// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Resamples one image into another of any size. Pixel centres are mapped
// onto each other (align_corners = false) and reads are clamped to the edge.

struct PushConstants {
    uint srcWidth;
    uint srcHeight;
    uint dstWidth;
    uint dstHeight;
    uint filterMode; // 0: nearest, 1: bilinear
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] [[vk::image_format("rgba8")]] RWTexture2D<float4> inputImage;
[[vk::binding(1, 0)]] [[vk::image_format("rgba8")]] WTexture2D<float4> outputImage;

float4 load(int2 p) {
    return inputImage.Load(clamp(p, int2(0), int2(pc.srcWidth - 1, pc.srcHeight - 1)));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    uint2 pixel = globalId.xy;
    if (pixel.x >= pc.dstWidth || pixel.y >= pc.dstHeight) {
        return;
    }

    float2 scale = float2(pc.srcWidth, pc.srcHeight) / float2(pc.dstWidth, pc.dstHeight);
    float2 src = (float2(pixel) + 0.5) * scale;

    float4 color;
    if (pc.filterMode == 0u) {
        color = load(int2(floor(src)));
    } else {
        float2 p = src - 0.5;
        int2 p0 = int2(floor(p));
        float2 f = p - float2(p0);
        float4 top = lerp(load(p0), load(p0 + int2(1, 0)), f.x);
        float4 bottom = lerp(load(p0 + int2(0, 1)), load(p0 + int2(1, 1)), f.x);
        color = lerp(top, bottom, f.y);
    }
    outputImage.Store(int2(pixel), color);
}
