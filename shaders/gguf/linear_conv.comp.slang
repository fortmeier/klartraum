// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Causal depthwise convolution (kernel 4) of the linear-attention input
// projection, followed by SiLU, for the params[0] active tokens in order.
// `state` holds each channel's previous three inputs (oldest first) across
// submissions and is advanced here. One invocation per channel.

struct PushConstants {
    uint channels;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> X;        // [T][channels]
[[vk::binding(1, 0)]] StructuredBuffer<float4> weights; // [channels][4], oldest tap first
[[vk::binding(2, 0)]] StructuredBuffer<uint> params;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> state;  // [3][channels]
[[vk::binding(4, 0)]] RWStructuredBuffer<float> Y;      // [T][channels]

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    const uint channel = globalId.x;
    if (channel >= pc.channels) return;
    const float4 w = weights[channel];
    float s0 = state[channel], s1 = state[pc.channels + channel], s2 = state[2u * pc.channels + channel];
    const uint tokens = params[0];
    for (uint t = 0u; t < tokens; ++t) {
        const float x = X[t * pc.channels + channel];
        const float y = w.x * s0 + w.y * s1 + w.z * s2 + w.w * x;
        Y[t * pc.channels + channel] = y / (1.0 + exp(-y));
        s0 = s1;
        s1 = s2;
        s2 = x;
    }
    state[channel] = s0;
    state[pc.channels + channel] = s1;
    state[2u * pc.channels + channel] = s2;
}
