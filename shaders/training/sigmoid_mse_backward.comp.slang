// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// The mean squared error of sigmoid outputs against targets: the logits'
// gradients, by automatic differentiation of sigmoidSquaredError() with the
// seed 1 / elementCount, and each element's squared error for the loss.
import differentiable;

struct PushConstants {
    uint elementCount;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> logits;
[[vk::binding(1, 0)]] StructuredBuffer<float> targets;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> logitGradients;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> squaredErrors;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    const uint i = globalId.x;
    if (i >= pc.elementCount) return;
    var logit = diffPair(logits[i]);
    bwd_diff(sigmoidSquaredError)(logit, targets[i], 1.0 / float(pc.elementCount));
    logitGradients[i] = logit.d;
    squaredErrors[i] = sigmoidSquaredError(logits[i], targets[i]);
}
