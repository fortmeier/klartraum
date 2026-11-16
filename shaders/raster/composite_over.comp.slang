// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Draws premultiplied front colours with coverage (e.g. Gaussians blended until a mesh's surface) over premultiplied
// back colours (e.g. the antialiased mesh): output = front + (1 - front coverage) x back, coverage included.

struct PushConstants {
    uint width;
    uint height;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float4> front;   // per pixel: rgb, coverage
[[vk::binding(1, 0)]] StructuredBuffer<float4> back;    // per pixel: rgb, coverage
[[vk::binding(2, 0)]] RWStructuredBuffer<float4> output; // per pixel

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    const uint pixel = globalId.x;
    if (pixel >= pc.width * pc.height) return;
    const float4 f = front[pixel];
    output[pixel] = f + (1.0 - f.a) * back[pixel];
}
