// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// softmax(query x keyᵀ) x value for heads up to 64 wide (see flash_attention.slang).
import flash_attention;

[[vk::binding(0, 0)]] StructuredBuffer<float> queryData;
[[vk::binding(1, 0)]] StructuredBuffer<float> keyData; // [batch][depth][keys]
[[vk::binding(2, 0)]] StructuredBuffer<float4> value4;
[[vk::binding(3, 0)]] RWStructuredBuffer<float4> output4;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 localId: SV_GroupThreadID, uint3 groupId: SV_GroupID) {
    flashAttention<1, false>(localId, groupId, queryData, keyData, value4, queryData, output4);
}
