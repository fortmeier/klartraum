// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// The leaky ReLU (slope 0.2 below zero) of every element (see differentiable.slang).
import differentiable;

struct PushConstants {
    uint elementCount;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> inputData;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> outputData;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    const uint i = globalId.x;
    if (i < pc.elementCount) outputData[i] = LeakyReluActivation.apply(inputData[i]);
}
