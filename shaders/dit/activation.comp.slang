// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// An activation of `count` values, in place: GELU with the tanh approximation (kind 0) or SiLU (kind 1).

struct PushConstants {
    uint count;
    uint kind;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] RWStructuredBuffer<float> X;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    const uint i = globalId.x + globalId.y * 65535u * 256u;
    if (i >= pc.count) return;
    const float x = X[i];
    if (pc.kind == 0u) {
        const float inner = 0.7978845608 * (x + 0.044715 * x * x * x);
        X[i] = 0.5 * x * (1.0 + tanh(inner));
    } else {
        X[i] = x / (1.0 + exp(-x));
    }
}
