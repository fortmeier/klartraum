// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Gradients of composite_over.comp.slang (output = front + (1 - front.a) back) from the output's gradients g:
//   dL/dfront.rgb = g.rgb,   dL/dfront.a = g.a - dot(g, back),   dL/dback = (1 - front.a) g.

struct PushConstants {
    uint width;
    uint height;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float4> front;             // per pixel: rgb, coverage
[[vk::binding(1, 0)]] StructuredBuffer<float4> back;              // per pixel: rgb, coverage
[[vk::binding(2, 0)]] StructuredBuffer<float4> outputGradients;   // per pixel: dL/d(rgb, coverage)
[[vk::binding(3, 0)]] RWStructuredBuffer<float4> frontGradients;  // per pixel, written
[[vk::binding(4, 0)]] RWStructuredBuffer<float4> backGradients;   // per pixel, written

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    const uint pixel = globalId.x;
    if (pixel >= pc.width * pc.height) return;
    const float4 f = front[pixel], b = back[pixel], g = outputGradients[pixel];
    frontGradients[pixel] = float4(g.rgb, g.a - dot(g, b));
    backGradients[pixel] = (1.0 - f.a) * g;
}
