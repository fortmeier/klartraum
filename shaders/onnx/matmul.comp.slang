// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import matmul;

[[vk::push_constant]] ConstantBuffer<MatMulPushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> lhs;
[[vk::binding(1, 0)]] StructuredBuffer<float> rhs;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> outputData;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 localId: SV_GroupThreadID, uint3 groupId: SV_GroupID) {
    matmul<IdentityEpilogue>(localId, groupId, pc, lhs, rhs, outputData);
}
