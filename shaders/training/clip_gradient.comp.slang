// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Gradient clipping by the global norm, as torch.nn.utils.clip_grad_norm_: every gradient times
// min(1, maxNorm / (norm + 1e-6)), the norm the square root of the running total of squared_norm.comp.slang.

struct PushConstants {
    uint count;
    float maxNorm;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] RWStructuredBuffer<float> gradient;
[[vk::binding(1, 0)]] StructuredBuffer<float> total;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 id: SV_DispatchThreadID) {
    if (id.x >= pc.count) return;
    const float factor = min(1.0, pc.maxNorm / (sqrt(total[0]) + 1e-6));
    gradient[id.x] *= factor;
}
