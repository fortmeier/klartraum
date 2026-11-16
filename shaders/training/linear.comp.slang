// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// A linear layer as nn.Linear: y[b][o] = sum_i x[b][i] W[o][i] + bias[o] + offset. One invocation per output.

struct PushConstants {
    uint batch;
    uint inputs;
    uint outputs;
    float offset;     // added to every output (1 for FiLM's scale heads)
    uint accumulate;  // backward input: 1 adds to the input gradient
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> x;      // [batch][inputs]
[[vk::binding(1, 0)]] StructuredBuffer<float> weight; // [outputs][inputs]
[[vk::binding(2, 0)]] StructuredBuffer<float> bias;   // [outputs]
[[vk::binding(3, 0)]] RWStructuredBuffer<float> y;    // [batch][outputs]

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID) {
    if (id.x >= pc.batch * pc.outputs) return;
    const uint b = id.x / pc.outputs, o = id.x % pc.outputs;
    float sum = bias[o] + pc.offset;
    for (uint i = 0u; i < pc.inputs; ++i) sum += x[b * pc.inputs + i] * weight[o * pc.inputs + i];
    y[id.x] = sum;
}
