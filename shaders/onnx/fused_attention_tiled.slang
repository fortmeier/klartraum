// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Tiled fused attention for a compile-time head width HEAD_DEPTH, where the
// query, key, and value depths are equal. Each invocation owns one query row
// and keeps it and its output accumulators in registers. The workgroup stages
// KEY_TILE keys and values in shared memory at a time, so every staged row is
// reused by QUERIES_PER_GROUP queries, and the online-softmax rescale runs once
// per KEY_BLOCK keys instead of once per key.
//
// Layouts: Q [batch, queryCount, depth], K [batch, depth, keyCount],
// V [batch, keyCount, depth], output [batch, queryCount, depth].
//
// Shaders run tiledAttention() from a QUERIES_PER_GROUP-wide workgroup and
// own the shared key and value tiles, KEY_TILE * HEAD_DEPTH floats each.
module fused_attention_tiled;

public static const uint QUERIES_PER_GROUP = 64;
static const uint KEY_BLOCK = 8;
static const float lowestScore = -3.402823466e+38;

public struct AttentionPushConstants {
    public uint batchCount;
    public uint queryCount;
    public uint keyCount;
    public uint queryDepth;
    public uint valueDepth;
};

// The shared key and value tiles. Both are stored key-major,
// [KEY_TILE][HEAD_DEPTH], so all invocations read the same address in the
// inner loops and the load is a broadcast.
public interface IAttentionTiles {
    static float key(uint index);
    static void setKey(uint index, float value);
    static float value(uint index);
    static void setValue(uint index, float value);
}

public void tiledAttention<let HEAD_DEPTH : uint, let KEY_TILE : uint, Tiles : IAttentionTiles>(
    uint localIndex, uint3 groupId, AttentionPushConstants pc, StructuredBuffer<float> queryData,
    StructuredBuffer<float> keyData, StructuredBuffer<float> valueData, RWStructuredBuffer<float> outputData) {
    const uint batch = groupId.y;
    const uint queryIndex = groupId.x * QUERIES_PER_GROUP + localIndex;
    const bool queryIsValid = queryIndex < pc.queryCount;

    float query[HEAD_DEPTH];
    float accumulated[HEAD_DEPTH];
    const uint queryOffset = (batch * pc.queryCount + queryIndex) * HEAD_DEPTH;
    for (uint depth = 0; depth < HEAD_DEPTH; ++depth) {
        query[depth] = queryIsValid ? queryData[queryOffset + depth] : 0.0;
        accumulated[depth] = 0.0;
    }
    float runningMaximum = lowestScore;
    float runningSum = 0.0;

    const uint keyBatchOffset = batch * HEAD_DEPTH * pc.keyCount;
    const uint valueBatchOffset = batch * pc.keyCount * HEAD_DEPTH;
    for (uint tileStart = 0; tileStart < pc.keyCount; tileStart += KEY_TILE) {
        // K is depth-major in memory; consecutive invocations read consecutive
        // keys of one depth row, which keeps the global loads coalesced.
        for (uint element = localIndex; element < KEY_TILE * HEAD_DEPTH; element += QUERIES_PER_GROUP) {
            const uint tileKey = element % KEY_TILE;
            const uint depth = element / KEY_TILE;
            const uint key = tileStart + tileKey;
            Tiles.setKey(tileKey * HEAD_DEPTH + depth,
                         key < pc.keyCount ? keyData[keyBatchOffset + depth * pc.keyCount + key] : 0.0);
        }
        // V rows are contiguous, so the tile is one contiguous range.
        for (uint element = localIndex; element < KEY_TILE * HEAD_DEPTH; element += QUERIES_PER_GROUP) {
            const uint key = tileStart + element / HEAD_DEPTH;
            Tiles.setValue(element,
                           key < pc.keyCount ? valueData[valueBatchOffset + tileStart * HEAD_DEPTH + element] : 0.0);
        }
        GroupMemoryBarrierWithGroupSync();

        for (uint blockStart = 0; blockStart < KEY_TILE; blockStart += KEY_BLOCK) {
            float scores[KEY_BLOCK];
            float blockMaximum = lowestScore;
            for (uint blockKey = 0; blockKey < KEY_BLOCK; ++blockKey) {
                const uint tileKey = blockStart + blockKey;
                float score = 0.0;
                for (uint depth = 0; depth < HEAD_DEPTH; ++depth) {
                    score += query[depth] * Tiles.key(tileKey * HEAD_DEPTH + depth);
                }
                // Keys past keyCount get the lowest score, so their softmax
                // weight is exactly zero once a real key has set the maximum.
                scores[blockKey] = tileStart + tileKey < pc.keyCount ? score : lowestScore;
                blockMaximum = max(blockMaximum, scores[blockKey]);
            }

            const float nextMaximum = max(runningMaximum, blockMaximum);
            const float previousScale = exp(runningMaximum - nextMaximum);
            runningSum *= previousScale;
            for (uint depth = 0; depth < HEAD_DEPTH; ++depth) accumulated[depth] *= previousScale;
            for (uint blockKey = 0; blockKey < KEY_BLOCK; ++blockKey) {
                const float weight = exp(scores[blockKey] - nextMaximum);
                runningSum += weight;
                const uint valueOffset = (blockStart + blockKey) * HEAD_DEPTH;
                for (uint depth = 0; depth < HEAD_DEPTH; ++depth) {
                    accumulated[depth] += weight * Tiles.value(valueOffset + depth);
                }
            }
            runningMaximum = nextMaximum;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (!queryIsValid) return;
    const float inverseSum = 1.0 / runningSum;
    for (uint depth = 0; depth < HEAD_DEPTH; ++depth) {
        outputData[queryOffset + depth] = accumulated[depth] * inverseSum;
    }
}
