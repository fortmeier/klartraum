// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// MatMul followed by ReLU² (Cosmos3's MLP activation): stores max(x, 0)^2.

import matmul;

[[vk::push_constant]] ConstantBuffer<MatMulPushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> lhs;
[[vk::binding(1, 0)]] StructuredBuffer<float> rhs;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> outputData;

struct ReluSquare : IMatMulEpilogue {
    static float apply(float value, uint index) {
        const float positive = max(value, 0.0);
        return positive * positive;
    }
};

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 localId: SV_GroupThreadID, uint3 groupId: SV_GroupID) {
    matmul<ReluSquare>(localId, groupId, pc, lhs, rhs, outputData);
}
