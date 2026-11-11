// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// FiLM: every channel of every sample scaled and shifted, y = x * scale[n][c] + shift[n][c]; with `shift` alone
// (scale bound to ones) also a per-sample bias. One invocation per element.

struct PushConstants {
    uint batch;
    uint channels;
    uint pixels;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> x;     // [batch][channels][pixels]
[[vk::binding(1, 0)]] StructuredBuffer<float> scale; // [batch][channels]
[[vk::binding(2, 0)]] StructuredBuffer<float> shift; // [batch][channels]
[[vk::binding(3, 0)]] RWStructuredBuffer<float> y;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 id: SV_DispatchThreadID) {
    if (id.x >= pc.batch * pc.channels * pc.pixels) return;
    const uint plane = id.x / pc.pixels;
    y[id.x] = x[id.x] * scale[plane] + shift[plane];
}
