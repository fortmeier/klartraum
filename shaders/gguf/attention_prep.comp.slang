// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Prepares the queries, keys and values of a gated full-attention layer for
// the params[0] active tokens at positions params[1] + t:
//  - queries (the first half of each head of the joint query/gate projection)
//    and keys are RMS-normalized per head, then rotated (NEOX RoPE on the
//    first `ropeDims` dimensions); queries go to Q, keys to the key cache;
//  - values are copied to the value cache.
// Workgroup (t, y): y < queryHeads is a query head, then the key heads, then
// the value heads. One invocation per head dimension (headDim <= 256).
import subgroup;

struct PushConstants {
    uint headDim;
    uint queryHeads;
    uint kvHeads;
    uint ropeDims;
    uint contextLength;
    float ropeBase;
    float epsilon;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> QG; // [T][queryHeads][2][headDim]
[[vk::binding(1, 0)]] StructuredBuffer<float> K;  // [T][kvHeads][headDim]
[[vk::binding(2, 0)]] StructuredBuffer<float> V;  // [T][kvHeads][headDim]
[[vk::binding(3, 0)]] StructuredBuffer<float> queryNorm;
[[vk::binding(4, 0)]] StructuredBuffer<float> keyNorm;
[[vk::binding(5, 0)]] StructuredBuffer<uint> params;
[[vk::binding(6, 0)]] RWStructuredBuffer<float> Q;          // [T][queryHeads][headDim]
[[vk::binding(7, 0)]] RWStructuredBuffer<float> keyCache;   // [kvHeads][context][headDim]
[[vk::binding(8, 0)]] RWStructuredBuffer<float> valueCache;

// Each subgroup's sum of squares.
groupshared float partial[8];
// The normalized head, read by the rotation of the paired dimension.
groupshared float normalized[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 localId: SV_GroupThreadID, uint3 groupId: SV_GroupID) {
    const uint token = groupId.x;
    if (token >= params[0]) return; // uniform per workgroup
    const uint position = params[1] + token;
    const uint head = groupId.y;
    const uint d = localId.x;
    const bool inHead = d < pc.headDim;

    if (head >= pc.queryHeads + pc.kvHeads) {
        const uint kvHead = head - pc.queryHeads - pc.kvHeads;
        if (inHead) {
            valueCache[(kvHead * pc.contextLength + position) * pc.headDim + d] =
                V[(token * pc.kvHeads + kvHead) * pc.headDim + d];
        }
        return;
    }

    const bool isQuery = head < pc.queryHeads;
    const uint kvHead = head - pc.queryHeads;
    float value = 0.0;
    if (inHead) {
        value = isQuery ? QG[((token * pc.queryHeads + head) * 2u) * pc.headDim + d]
                        : K[(token * pc.kvHeads + kvHead) * pc.headDim + d];
    }
    const float square = WaveActiveSum(value * value);
    if (WaveIsFirstLane()) partial[subgroupId()] = square;
    GroupMemoryBarrierWithGroupSync();
    float sum = 0.0;
    for (uint s = 0u; s < subgroupCount(); ++s) sum += partial[s];
    const float weight = inHead ? (isQuery ? queryNorm[d] : keyNorm[d]) : 0.0;
    normalized[d] = value * rsqrt(sum / float(pc.headDim) + pc.epsilon) * weight;
    GroupMemoryBarrierWithGroupSync();
    if (!inHead) return;

    float rotated = normalized[d];
    if (d < pc.ropeDims) {
        const uint half_ = pc.ropeDims / 2u;
        const uint pair = d % half_;
        const float frequency = pow(pc.ropeBase, -2.0 * float(pair) / float(pc.ropeDims));
        const float angle = float(position) * frequency;
        const float c = cos(angle), s = sin(angle);
        rotated = d < half_ ? normalized[d] * c - normalized[d + half_] * s
                            : normalized[d] * c + normalized[d - half_] * s;
    }
    if (isQuery) {
        Q[(token * pc.queryHeads + head) * pc.headDim + d] = rotated;
    } else {
        keyCache[(kvHead * pc.contextLength + position) * pc.headDim + d] = rotated;
    }
}
