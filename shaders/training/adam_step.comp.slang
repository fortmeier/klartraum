// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// One Adam step (Kingma & Ba 2015) on a parameter tensor, in place. The step
// number t comes from a counter on the GPU (step_counter.comp.slang), so a
// pre-recorded training graph applies the right bias corrections every
// submission, and the learning rate schedule: an exponential decay to
// learningRate x finalFactor over decaySteps steps (none for decaySteps 0).

struct PushConstants {
    uint elementCount;
    float learningRate;
    float beta1;
    float beta2;
    float epsilon;
    uint decaySteps;
    float finalFactor;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] RWStructuredBuffer<float> parameters;
[[vk::binding(1, 0)]] StructuredBuffer<float> gradients;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> firstMoments;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> secondMoments;
[[vk::binding(4, 0)]] StructuredBuffer<uint> step; // t, from 1

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    const uint i = globalId.x;
    if (i >= pc.elementCount) return;
    const float g = gradients[i];
    const float m = pc.beta1 * firstMoments[i] + (1.0 - pc.beta1) * g;
    const float v = pc.beta2 * secondMoments[i] + (1.0 - pc.beta2) * g * g;
    firstMoments[i] = m;
    secondMoments[i] = v;
    const float t = float(step[0]);
    const float mHat = m / (1.0 - pow(pc.beta1, t));
    const float vHat = v / (1.0 - pow(pc.beta2, t));
    const float learningRate =
        pc.decaySteps > 0 ? pc.learningRate * pow(pc.finalFactor, min(t / float(pc.decaySteps), 1.0)) : pc.learningRate;
    parameters[i] -= learningRate * mHat / (sqrt(vHat) + pc.epsilon);
}
