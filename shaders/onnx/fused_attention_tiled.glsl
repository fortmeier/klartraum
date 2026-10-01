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

#ifndef HEAD_DEPTH
#error "HEAD_DEPTH must be defined before including fused_attention_tiled.glsl"
#endif

#define QUERIES_PER_GROUP 64
#ifndef KEY_TILE
#define KEY_TILE 32
#endif
#define KEY_BLOCK 8

layout(local_size_x = QUERIES_PER_GROUP, local_size_y = 1, local_size_z = 1) in;
layout(push_constant) uniform PushConstants {
    uint batchCount;
    uint queryCount;
    uint keyCount;
    uint queryDepth;
    uint valueDepth;
} pc;
layout(set = 0, binding = 0) restrict readonly buffer QueryBuffer { float queryData[]; };
layout(set = 0, binding = 1) restrict readonly buffer KeyBuffer { float keyData[]; };
layout(set = 0, binding = 2) restrict readonly buffer ValueBuffer { float valueData[]; };
layout(set = 0, binding = 3) restrict writeonly buffer OutputBuffer { float outputData[]; };

// Both tiles are stored key-major, [KEY_TILE][HEAD_DEPTH], so all invocations
// read the same address in the inner loops and the load is a broadcast.
shared float sharedKey[KEY_TILE * HEAD_DEPTH];
shared float sharedValue[KEY_TILE * HEAD_DEPTH];

const float lowestScore = -3.402823466e+38;

void main() {
    const uint localIndex = gl_LocalInvocationID.x;
    const uint batch = gl_WorkGroupID.y;
    const uint queryIndex = gl_WorkGroupID.x * QUERIES_PER_GROUP + localIndex;
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
            sharedKey[tileKey * HEAD_DEPTH + depth] =
                key < pc.keyCount ? keyData[keyBatchOffset + depth * pc.keyCount + key] : 0.0;
        }
        // V rows are contiguous, so the tile is one contiguous range.
        for (uint element = localIndex; element < KEY_TILE * HEAD_DEPTH; element += QUERIES_PER_GROUP) {
            const uint key = tileStart + element / HEAD_DEPTH;
            sharedValue[element] =
                key < pc.keyCount ? valueData[valueBatchOffset + tileStart * HEAD_DEPTH + element] : 0.0;
        }
        barrier();

        for (uint blockStart = 0; blockStart < KEY_TILE; blockStart += KEY_BLOCK) {
            float scores[KEY_BLOCK];
            float blockMaximum = lowestScore;
            for (uint blockKey = 0; blockKey < KEY_BLOCK; ++blockKey) {
                const uint tileKey = blockStart + blockKey;
                float score = 0.0;
                for (uint depth = 0; depth < HEAD_DEPTH; ++depth) {
                    score += query[depth] * sharedKey[tileKey * HEAD_DEPTH + depth];
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
                    accumulated[depth] += weight * sharedValue[valueOffset + depth];
                }
            }
            runningMaximum = nextMaximum;
        }
        barrier();
    }

    if (!queryIsValid) return;
    const float inverseSum = 1.0 / runningSum;
    for (uint depth = 0; depth < HEAD_DEPTH; ++depth) {
        outputData[queryOffset + depth] = accumulated[depth] * inverseSum;
    }
}
