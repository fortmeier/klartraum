// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Multi-head attention of all `tokens` queries over all tokens' keys (or, with `causal`, over the keys up to their
// own), e.g. the joint text and image attention of a diffusion transformer, with an online softmax (flash attention):
//   out[t][h] = sum_s softmax_s(q[t][h] . k[s][h] / sqrt(128)) v[s][h].
// Queries, keys, values and outputs are [tokens][heads][128].
//
// A workgroup of 128 invocations computes 16 queries of one head: 8 invocations per query, each holding 16 of its 128
// values. Keys and values pass through shared memory in blocks of 16; a query's 8 invocations sum their partial dot
// products with subgroup shuffles (they are 8 consecutive lanes).

struct PushConstants {
    uint tokens;
    uint heads;
    uint causal;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float4> Q;
[[vk::binding(1, 0)]] StructuredBuffer<float4> K;
[[vk::binding(2, 0)]] StructuredBuffer<float4> V;
[[vk::binding(3, 0)]] RWStructuredBuffer<float4> O;

static const uint DIM = 128u;
static const uint QUERIES = 16u; // per workgroup
static const uint LANES = 8u;    // per query
static const uint KEYS = 16u;    // per block

groupshared float4 keys[KEYS][DIM / 4u];
groupshared float4 values[KEYS][DIM / 4u];

// The sum of `value` over the 8 lanes of this query.
float querySum(float value) {
    value += WaveReadLaneAt(value, WaveGetLaneIndex() ^ 1u);
    value += WaveReadLaneAt(value, WaveGetLaneIndex() ^ 2u);
    value += WaveReadLaneAt(value, WaveGetLaneIndex() ^ 4u);
    return value;
}

[shader("compute")]
[numthreads(QUERIES * LANES, 1, 1)]
void main(uint3 groupId: SV_GroupID, uint local: SV_GroupIndex) {
    const uint head = groupId.y;
    const uint query = groupId.x * QUERIES + local / LANES, part = local % LANES; // part: values 16 part on
    const bool active = query < pc.tokens;
    const uint row = (min(query, pc.tokens - 1u) * pc.heads + head) * (DIM / 4u) + 4u * part;
    const float scale = 1.0 / sqrt(float(DIM));
    float4 q[4], o[4];
    [ForceUnroll]
    for (uint k = 0u; k < 4u; ++k) {
        q[k] = Q[row + k] * scale;
        o[k] = 0.0;
    }
    float maximum = -1e30, total = 0.0;
    // With `causal`, the workgroup's keys end after its last query.
    const uint lastQuery = min(groupId.x * QUERIES + QUERIES, pc.tokens);
    const uint keyCount = pc.causal != 0u ? lastQuery : pc.tokens;
    for (uint start = 0u; start < keyCount; start += KEYS) {
        // 128 invocations load 16 keys and 16 values of 32 float4 each: 4 + 4 per invocation.
        [ForceUnroll]
        for (uint k = 0u; k < 4u; ++k) {
            const uint index = local + 128u * k, key = index / (DIM / 4u), column = index % (DIM / 4u);
            const uint source = (min(start + key, pc.tokens - 1u) * pc.heads + head) * (DIM / 4u) + column;
            keys[key][column] = K[source];
            values[key][column] = V[source];
        }
        GroupMemoryBarrierWithGroupSync();
        const uint blockKeys = min(KEYS, keyCount - start);
        for (uint key = 0u; key < blockKeys; ++key) {
            float partial = 0.0;
            [ForceUnroll]
            for (uint k = 0u; k < 4u; ++k) partial += dot(q[k], keys[key][4u * part + k]);
            const float score = querySum(partial);
            if (pc.causal != 0u && start + key > query) continue;
            const float next = max(maximum, score);
            const float rescale = exp(maximum - next), weight = exp(score - next);
            total = total * rescale + weight;
            [ForceUnroll]
            for (uint k = 0u; k < 4u; ++k) o[k] = o[k] * rescale + weight * values[key][4u * part + k];
            maximum = next;
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (active) {
        [ForceUnroll]
        for (uint k = 0u; k < 4u; ++k) O[row + k] = o[k] / total;
    }
}
