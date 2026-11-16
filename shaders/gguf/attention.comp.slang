// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Causal grouped-query attention of the params[0] active tokens against the
// key/value caches, gated by the sigmoid of the query projection's gate half:
//   out = softmax(q K^T * scale) V * sigmoid(gate)
// Token t (position params[1] + t) attends cache positions 0 .. params[1] + t.
// Workgroup (t, head); its four subgroups take every fourth key with a
// running softmax each, then merge. Each lane owns headDim / 32 dimensions.
import subgroup;

static const uint SUBGROUPS = 4;
static const uint MAX_DIMS_PER_LANE = 8;
static const float NEGATIVE_INFINITY = asfloat(0xff800000u);

struct PushConstants {
    uint headDim;
    uint queryHeads;
    uint kvHeads;
    uint contextLength;
    float scale;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> Q;          // [T][queryHeads][headDim]
[[vk::binding(1, 0)]] StructuredBuffer<float> keyCache;   // [kvHeads][context][headDim]
[[vk::binding(2, 0)]] StructuredBuffer<float> valueCache;
[[vk::binding(3, 0)]] StructuredBuffer<float> QG;         // [T][queryHeads][2][headDim]
[[vk::binding(4, 0)]] StructuredBuffer<uint> params;
[[vk::binding(5, 0)]] RWStructuredBuffer<float> O;        // [T][queryHeads][headDim]

// The running maximum of each subgroup's softmax.
groupshared float maxima[SUBGROUPS];
// The running sum of each subgroup's softmax.
groupshared float sums[SUBGROUPS];
// Each subgroup's weighted value sum.
groupshared float accumulated[SUBGROUPS][256];

[shader("compute")]
[numthreads(32 * SUBGROUPS, 1, 1)]
void main(uint3 localId: SV_GroupThreadID, uint3 groupId: SV_GroupID) {
    const uint token = groupId.x;
    if (token >= params[0]) return;
    const uint head = groupId.y;
    const uint kvHead = head / (pc.queryHeads / pc.kvHeads);
    const uint lastKey = params[1] + token;
    const uint lane = WaveGetLaneIndex(), subgroup = subgroupId();
    const uint dims = pc.headDim / 32u;
    const uint firstDim = lane * dims;

    float q[MAX_DIMS_PER_LANE];
    float acc[MAX_DIMS_PER_LANE];
    const uint queryBase = (token * pc.queryHeads + head) * pc.headDim + firstDim;
    for (uint i = 0u; i < MAX_DIMS_PER_LANE; ++i) {
        q[i] = i < dims ? Q[queryBase + i] : 0.0;
        acc[i] = 0.0;
    }

    float runningMax = NEGATIVE_INFINITY;
    float runningSum = 0.0;
    const uint cacheBase = kvHead * pc.contextLength * pc.headDim + firstDim;
    for (uint key = subgroup; key <= lastKey; key += SUBGROUPS) {
        const uint base = cacheBase + key * pc.headDim;
        float product = 0.0;
        for (uint i = 0u; i < MAX_DIMS_PER_LANE; ++i) {
            if (i < dims) product += q[i] * keyCache[base + i];
        }
        const float score = WaveActiveSum(product) * pc.scale;
        const float newMax = max(runningMax, score);
        const float correction = exp(runningMax - newMax);
        const float weight = exp(score - newMax);
        runningSum = runningSum * correction + weight;
        for (uint i = 0u; i < MAX_DIMS_PER_LANE; ++i) {
            if (i < dims) acc[i] = acc[i] * correction + weight * valueCache[base + i];
        }
        runningMax = newMax;
    }

    if (lane == 0u) {
        maxima[subgroup] = runningMax;
        sums[subgroup] = runningSum;
    }
    for (uint i = 0u; i < dims; ++i) accumulated[subgroup][firstDim + i] = acc[i];
    GroupMemoryBarrierWithGroupSync();

    float globalMax = NEGATIVE_INFINITY;
    for (uint s = 0u; s < SUBGROUPS; ++s) globalMax = max(globalMax, maxima[s]);
    float total = 0.0;
    float factors[SUBGROUPS];
    for (uint s = 0u; s < SUBGROUPS; ++s) {
        // Subgroups without keys have sum 0 and maximum -inf: factor 0.
        factors[s] = sums[s] > 0.0 ? exp(maxima[s] - globalMax) : 0.0;
        total += sums[s] * factors[s];
    }
    const uint gateBase = ((token * pc.queryHeads + head) * 2u + 1u) * pc.headDim;
    for (uint d = localId.x; d < pc.headDim; d += 32u * SUBGROUPS) {
        float value = 0.0;
        for (uint s = 0u; s < SUBGROUPS; ++s) value += accumulated[s][d] * factors[s];
        const float gate = QG[gateBase + d];
        O[(token * pc.queryHeads + head) * pc.headDim + d] = value / total / (1.0 + exp(-gate));
    }
}
