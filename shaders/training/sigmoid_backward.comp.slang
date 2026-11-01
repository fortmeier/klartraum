// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// The input gradient of the sigmoid activation (see differentiable.slang).
import differentiable;

struct PushConstants {
    uint elementCount;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> preActivation;
[[vk::binding(1, 0)]] StructuredBuffer<float> outputGradient;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> inputGradient;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    const uint i = globalId.x;
    if (i < pc.elementCount) inputGradient[i] = activationGradient<SigmoidActivation>(preActivation[i], outputGradient[i]);
}
