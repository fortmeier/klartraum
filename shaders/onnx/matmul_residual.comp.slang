// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// MatMul followed by a residual Add of the output's shape.

import matmul;

[[vk::push_constant]] ConstantBuffer<MatMulPushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> lhs;
[[vk::binding(1, 0)]] StructuredBuffer<float> rhs;
[[vk::binding(2, 0)]] StructuredBuffer<float> residualData;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> outputData;

struct Residual : IMatMulEpilogue {
    static float apply(float value, uint index) { return value + residualData[index]; }
};

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 localId: SV_GroupThreadID, uint3 groupId: SV_GroupID) {
    matmul<Residual>(localId, groupId, pc, lhs, rhs, outputData);
}
