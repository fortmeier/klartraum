// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Rotary position embedding of queries or keys [tokens][heads][128], in place: each pair of consecutive values (2i,
// 2i + 1) is turned as a complex number by the angle whose cosine and sine the token's row of the tables holds (64 per
// token), as Qwen-Image's apply_rotary_emb_qwen with use_real=False.

struct PushConstants {
    uint tokens;
    uint heads;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> cosines; // [tokens][64]
[[vk::binding(1, 0)]] StructuredBuffer<float> sines;   // [tokens][64]
[[vk::binding(2, 0)]] RWStructuredBuffer<float2> X;    // [tokens][heads][64] pairs

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 groupId: SV_GroupID, uint local: SV_GroupIndex) {
    const uint token = groupId.x, head = groupId.y;
    if (token >= pc.tokens) return;
    const float c = cosines[token * 64u + local], s = sines[token * 64u + local];
    const uint index = (token * pc.heads + head) * 64u + local;
    const float2 x = X[index];
    X[index] = float2(x.x * c - x.y * s, x.x * s + x.y * c);
}
