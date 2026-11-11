// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// The input gradient of linear.comp.slang: dx[b][i] = sum_o dy[b][o] W[o][i]; with accumulate added to dx (several
// heads reading one input). One invocation per input element.

struct PushConstants {
    uint batch;
    uint inputs;
    uint outputs;
    float offset;     // added to every output (1 for FiLM's scale heads)
    uint accumulate;  // backward input: 1 adds to the input gradient
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> outputGradient; // [batch][outputs]
[[vk::binding(1, 0)]] StructuredBuffer<float> weight;         // [outputs][inputs]
[[vk::binding(2, 0)]] RWStructuredBuffer<float> inputGradient; // [batch][inputs]

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID) {
    if (id.x >= pc.batch * pc.inputs) return;
    const uint b = id.x / pc.inputs, i = id.x % pc.inputs;
    float sum = 0.0;
    for (uint o = 0u; o < pc.outputs; ++o) sum += outputGradient[b * pc.outputs + o] * weight[o * pc.inputs + i];
    inputGradient[id.x] = pc.accumulate != 0u ? inputGradient[id.x] + sum : sum;
}
