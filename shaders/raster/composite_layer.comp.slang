// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Draws a layer of premultiplied colours with coverage (e.g. an antialiased mesh) over an image:
// image = layer.rgb + (1 - layer.a) * image.

struct PushConstants {
    uint width;
    uint height;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] [[vk::image_format("rgba8")]] RWTexture2D<float4> image;
[[vk::binding(1, 0)]] StructuredBuffer<float4> layer; // per pixel, row-major

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    if (globalId.x >= pc.width || globalId.y >= pc.height) return;
    const float4 top = layer[globalId.y * pc.width + globalId.x];
    if (top.a <= 0.0) return;
    const float4 below = image[globalId.xy];
    image[globalId.xy] = float4(clamp(top.rgb + (1.0 - top.a) * below.rgb, 0.0, 1.0), 1.0);
}
