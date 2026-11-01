// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// softmax(query x keyᵀ + bias) x value for 128-wide heads (see flash_attention.slang).
import flash_attention;

[[vk::binding(0, 0)]] StructuredBuffer<float> queryData;
[[vk::binding(1, 0)]] StructuredBuffer<float> keyData; // [batch][depth][keys]
[[vk::binding(2, 0)]] StructuredBuffer<float4> value4;
[[vk::binding(3, 0)]] StructuredBuffer<float> biasData;
[[vk::binding(4, 0)]] RWStructuredBuffer<float4> output4;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 localId: SV_GroupThreadID, uint3 groupId: SV_GroupID) {
    flashAttention<2, true>(localId, groupId, queryData, keyData, value4, biasData, output4);
}
