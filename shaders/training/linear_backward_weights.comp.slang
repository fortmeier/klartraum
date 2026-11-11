// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// The weight and bias gradients of linear.comp.slang: dW[o][i] = sum_b dy[b][o] x[b][i], dbias[o] = sum_b dy[b][o].
// One invocation per weight, then one per bias.

struct PushConstants {
    uint batch;
    uint inputs;
    uint outputs;
    float offset;     // added to every output (1 for FiLM's scale heads)
    uint accumulate;  // backward input: 1 adds to the input gradient
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> x;              // [batch][inputs]
[[vk::binding(1, 0)]] StructuredBuffer<float> outputGradient; // [batch][outputs]
[[vk::binding(2, 0)]] RWStructuredBuffer<float> weightGradient; // [outputs][inputs]
[[vk::binding(3, 0)]] RWStructuredBuffer<float> biasGradient;   // [outputs]

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID) {
    const uint weights = pc.outputs * pc.inputs;
    if (id.x >= weights + pc.outputs) return;
    float sum = 0.0;
    if (id.x < weights) {
        const uint o = id.x / pc.inputs, i = id.x % pc.inputs;
        for (uint b = 0u; b < pc.batch; ++b) sum += outputGradient[b * pc.outputs + o] * x[b * pc.inputs + i];
        weightGradient[id.x] = sum;
    } else {
        const uint o = id.x - weights;
        for (uint b = 0u; b < pc.batch; ++b) sum += outputGradient[b * pc.outputs + o];
        biasGradient[o] = sum;
    }
}
