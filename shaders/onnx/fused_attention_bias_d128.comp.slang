// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// softmax(query x keyᵀ + bias) x value for 128-wide heads, without storing the
// score matrix (FlashAttention-style). A workgroup handles 64 queries of one
// batch-head and walks the keys in tiles of 64:
//  1. scores S[64 x 64] = Q x Kᵀ, register-blocked like MatMul: invocation
//     (tx, ty) owns queries 4 ty .. 4 ty + 3 and keys 4 tx .. 4 tx + 3; query
//     and key slices are staged 32 depth components at a time;
//  2. bias, masking, and the online softmax on the scores in registers; the
//     16 invocations sharing a query row (one half of a subgroup) combine
//     their row maxima and sums with shuffles;
//  3. O[64 x 128] += P x V with P in shared memory: each invocation owns the
//     output rows of its queries and columns 4 tx .. and 64 + 4 tx .., so the
//     per-row rescale stays in registers.
// Key tiles whose bias masks every (query, key) pair (text padding, the causal
// upper triangle) are skipped: their weights exp(bias + score - max) are
// exactly zero in float32 for biases at or below MASKED.
// One shared pool is reused by the phases (at most 24 KiB at a time).
import subgroup;

struct PushConstants {
    uint batchCount;
    uint heads;
    uint queryCount;
    uint keyCount;
    uint biasStrideBatch;
    uint biasStrideHead;
    uint biasStrideQuery;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> queryData;
[[vk::binding(1, 0)]] StructuredBuffer<float> keyData; // [batch][128][keys]
[[vk::binding(2, 0)]] StructuredBuffer<float4> value4;
[[vk::binding(3, 0)]] StructuredBuffer<float> biasData;
[[vk::binding(4, 0)]] RWStructuredBuffer<float4> output4;

static const uint DEPTH = 128;
static const uint QUERIES = 64;
static const uint KEYS = 64;
static const uint CHUNK = 32;      // depth components per staged query/key slice
static const uint VALUE_KEYS = 16; // keys per staged value slice
static const float LOWEST = -3.402823466e+38;
static const float MASKED = -1.0e6;

// Staging memory shared by the phases.
// Phase 1: queryChunk [CHUNK][QUERIES / 4] at 0, keyChunk [CHUNK][KEYS / 4] at 512.
// Phase 3: probabilities [KEYS][QUERIES / 4] at 0, valueChunk [VALUE_KEYS][DEPTH / 4] at 1024.
groupshared float4 pool[1536];
// The largest bias each subgroup sees in the current key tile.
groupshared float subgroupBiasMaximum[8];

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 localId: SV_GroupThreadID, uint3 groupId: SV_GroupID) {
    const uint tx = localId.x;
    const uint ty = localId.y;
    const uint threadIndex = ty * 16 + tx;
    const uint queryBase = groupId.x * QUERIES;
    const uint batch = groupId.y;
    const uint queryBatchOffset = batch * pc.queryCount * DEPTH;
    const uint keyBatchOffset = batch * DEPTH * pc.keyCount;
    const uint valueBatchVec4 = batch * pc.keyCount * (DEPTH / 4);
    const uint biasBase = (batch / pc.heads) * pc.biasStrideBatch + (batch % pc.heads) * pc.biasStrideHead;

    // Staging assignments: 8 query components (query threadIndex / 4, depth (threadIndex % 4) * 8 ..)
    // and 8 key components (depth threadIndex / 8, keys (threadIndex % 8) * 8 ..) per chunk.
    const uint stageQuery = threadIndex / 4;
    const uint stageQueryDepth = (threadIndex % 4) * 8;
    const uint stageKeyDepth = threadIndex / 8;
    const uint stageKey = (threadIndex % 8) * 8;

    float4 outputRows[4][2];
    float runningMaximum[4];
    float runningSum[4];
    [unroll]
    for (uint i = 0; i < 4; ++i) {
        outputRows[i][0] = float4(0.0);
        outputRows[i][1] = float4(0.0);
        runningMaximum[i] = LOWEST;
        runningSum[i] = 0.0;
    }

    for (uint keyBase = 0; keyBase < pc.keyCount; keyBase += KEYS) {
        // 0. This invocation's biases (keys past the end count as masked); skip fully masked tiles.
        float4 bias[4];
        float biasMaximum = LOWEST;
        [unroll]
        for (uint i = 0; i < 4; ++i) {
            const uint query = queryBase + ty * 4 + i;
            const uint biasRow = biasBase + (query < pc.queryCount ? query : 0) * pc.biasStrideQuery;
            [unroll]
            for (uint j = 0; j < 4; ++j) {
                const uint key = keyBase + tx * 4 + j;
                bias[i][j] = key < pc.keyCount ? biasData[biasRow + key] : LOWEST;
                if (query < pc.queryCount) biasMaximum = max(biasMaximum, bias[i][j]);
            }
        }
        biasMaximum = WaveActiveMax(biasMaximum);
        if (WaveIsFirstLane()) subgroupBiasMaximum[subgroupId()] = biasMaximum;
        GroupMemoryBarrierWithGroupSync();
        float tileBiasMaximum = LOWEST;
        [unroll]
        for (uint s = 0; s < 8; ++s) tileBiasMaximum = max(tileBiasMaximum, subgroupBiasMaximum[s]);
        GroupMemoryBarrierWithGroupSync();
        if (tileBiasMaximum <= MASKED) continue;

        // 1. Scores for this key tile.
        float4 scores[4] = { float4(0.0), float4(0.0), float4(0.0), float4(0.0) };
        for (uint depthBase = 0; depthBase < DEPTH; depthBase += CHUNK) {
            const uint query = queryBase + stageQuery;
            [unroll]
            for (uint i = 0; i < 8; ++i) {
                const uint depth = stageQueryDepth + i;
                pool[depth * (QUERIES / 4) + stageQuery / 4][stageQuery % 4] =
                    query < pc.queryCount ? queryData[queryBatchOffset + query * DEPTH + depthBase + depth] : 0.0;
            }
            [unroll]
            for (uint i = 0; i < 8; ++i) {
                const uint key = keyBase + stageKey + i;
                pool[512 + stageKeyDepth * (KEYS / 4) + (stageKey + i) / 4][(stageKey + i) % 4] =
                    key < pc.keyCount ? keyData[keyBatchOffset + (depthBase + stageKeyDepth) * pc.keyCount + key] : 0.0;
            }
            GroupMemoryBarrierWithGroupSync();
            [unroll]
            for (uint d = 0; d < CHUNK; ++d) {
                const float4 q = pool[d * (QUERIES / 4) + ty];
                const float4 k = pool[512 + d * (KEYS / 4) + tx];
                scores[0] += q.x * k;
                scores[1] += q.y * k;
                scores[2] += q.z * k;
                scores[3] += q.w * k;
            }
            GroupMemoryBarrierWithGroupSync();
        }

        // 2. Bias, masking, and the online softmax; rows are shared by the 16 tx of one ty.
        float rescale[4];
        [unroll]
        for (uint i = 0; i < 4; ++i) {
            float tileMaximum = LOWEST;
            [unroll]
            for (uint j = 0; j < 4; ++j) {
                const uint key = keyBase + tx * 4 + j;
                scores[i][j] = key < pc.keyCount ? scores[i][j] + bias[i][j] : LOWEST;
                tileMaximum = max(tileMaximum, scores[i][j]);
            }
            tileMaximum = max(tileMaximum, subgroupShuffleXor(tileMaximum, 1));
            tileMaximum = max(tileMaximum, subgroupShuffleXor(tileMaximum, 2));
            tileMaximum = max(tileMaximum, subgroupShuffleXor(tileMaximum, 4));
            tileMaximum = max(tileMaximum, subgroupShuffleXor(tileMaximum, 8));
            const float newMaximum = max(runningMaximum[i], tileMaximum);
            rescale[i] = exp(runningMaximum[i] - newMaximum);
            float tileSum = 0.0;
            [unroll]
            for (uint j = 0; j < 4; ++j) {
                const uint key = keyBase + tx * 4 + j;
                // Every tile holds at least one real key, so newMaximum is finite here.
                scores[i][j] = key < pc.keyCount ? exp(scores[i][j] - newMaximum) : 0.0;
                tileSum += scores[i][j];
            }
            tileSum += subgroupShuffleXor(tileSum, 1);
            tileSum += subgroupShuffleXor(tileSum, 2);
            tileSum += subgroupShuffleXor(tileSum, 4);
            tileSum += subgroupShuffleXor(tileSum, 8);
            runningSum[i] = runningSum[i] * rescale[i] + tileSum;
            runningMaximum[i] = newMaximum;
            outputRows[i][0] *= rescale[i];
            outputRows[i][1] *= rescale[i];
        }
        // Probabilities as [key][query / 4] for the value product.
        [unroll]
        for (uint j = 0; j < 4; ++j) {
            pool[(tx * 4 + j) * (QUERIES / 4) + ty] = float4(scores[0][j], scores[1][j], scores[2][j], scores[3][j]);
        }

        // 3. Output += probabilities x values, values staged 16 keys at a time.
        for (uint valueBase = 0; valueBase < KEYS; valueBase += VALUE_KEYS) {
            [unroll]
            for (uint r = 0; r < 2; ++r) {
                const uint element = threadIndex + 256 * r;
                const uint k = element / (DEPTH / 4);
                const uint column = element % (DEPTH / 4);
                const uint key = keyBase + valueBase + k;
                pool[1024 + k * (DEPTH / 4) + column] =
                    key < pc.keyCount ? value4[valueBatchVec4 + key * (DEPTH / 4) + column] : float4(0.0);
            }
            GroupMemoryBarrierWithGroupSync();
            [unroll]
            for (uint k = 0; k < VALUE_KEYS; ++k) {
                const float4 p = pool[(valueBase + k) * (QUERIES / 4) + ty];
                const float4 v0 = pool[1024 + k * (DEPTH / 4) + tx];
                const float4 v1 = pool[1024 + k * (DEPTH / 4) + 16 + tx];
                [unroll]
                for (uint i = 0; i < 4; ++i) {
                    outputRows[i][0] += p[i] * v0;
                    outputRows[i][1] += p[i] * v1;
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }

    [unroll]
    for (uint i = 0; i < 4; ++i) {
        const uint query = queryBase + ty * 4 + i;
        if (query >= pc.queryCount) continue;
        const float normalizer = 1.0 / runningSum[i];
        const uint row = (batch * pc.queryCount + query) * (DEPTH / 4);
        output4[row + tx] = outputRows[i][0] * normalizer;
        output4[row + 16 + tx] = outputRows[i][1] * normalizer;
    }
}
