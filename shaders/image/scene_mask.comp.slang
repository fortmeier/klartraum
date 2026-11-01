// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// The scene mask: white where the depth buffer holds a surface, black where it holds +inf.

struct PushConstants {
    uint width;
    uint height;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> depth;
[[vk::binding(1, 0)]] [[vk::image_format("rgba8")]] WTexture2D<float4> mask;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    uint2 pixel = globalId.xy;
    if (pixel.x >= pc.width || pixel.y >= pc.height) {
        return;
    }
    bool surface = !isinf(depth[pixel.y * pc.width + pixel.x]);
    mask.Store(int2(pixel), surface ? float4(1.0) : float4(0.0, 0.0, 0.0, 1.0));
}
