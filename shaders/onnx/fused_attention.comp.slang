// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Eight subgroups compute eight query rows together. Key and value rows are
// staged once per workgroup and shared by all eight queries. Online softmax
// avoids ever storing the quadratic score matrix.

struct PushConstants {
    uint batchCount;
    uint queryCount;
    uint keyCount;
    uint queryDepth;
    uint valueDepth;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> queryData;
[[vk::binding(1, 0)]] StructuredBuffer<float> keyData;
[[vk::binding(2, 0)]] StructuredBuffer<float> valueData;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> outputData;

// The key row of the current key, shared by the eight queries.
groupshared float sharedKey[512];
// The value row of the current key, shared by the eight queries.
groupshared float sharedValue[512];

[shader("compute")]
[numthreads(32, 8, 1)]
void main(uint3 localId: SV_GroupThreadID, uint3 groupId: SV_GroupID) {
    const uint lane = WaveGetLaneIndex();
    const uint laneCount = WaveGetLaneCount();
    const uint batch = groupId.y;
    const uint queryIndex = groupId.x * 8 + localId.y;
    const bool queryIsValid = queryIndex < pc.queryCount;
    const uint workgroupLane = localId.y * 32 + localId.x;

    // SD1.5 UNet head widths are at most 160 and the VAE attention width is
    // 512. Sixteen strided accumulators cover both without dynamic arrays.
    float accumulated[16];
    for (uint slot = 0; slot < 16; ++slot) accumulated[slot] = 0.0;
    float runningMaximum = -3.402823466e+38;
    float runningSum = 0.0;

    const uint queryOffset = (batch * pc.queryCount + queryIndex) * pc.queryDepth;
    const uint keyBatchOffset = batch * pc.queryDepth * pc.keyCount;
    const uint valueBatchOffset = batch * pc.keyCount * pc.valueDepth;
    for (uint keyIndex = 0; keyIndex < pc.keyCount; ++keyIndex) {
        for (uint depth = workgroupLane; depth < pc.queryDepth; depth += 256) {
            sharedKey[depth] = keyData[keyBatchOffset + depth * pc.keyCount + keyIndex];
        }
        for (uint depth = workgroupLane; depth < pc.valueDepth; depth += 256) {
            sharedValue[depth] = valueData[valueBatchOffset + keyIndex * pc.valueDepth + depth];
        }
        GroupMemoryBarrierWithGroupSync();

        float partialScore = 0.0;
        for (uint depth = lane; queryIsValid && depth < pc.queryDepth; depth += laneCount) {
            partialScore += queryData[queryOffset + depth] * sharedKey[depth];
        }
        const float score = WaveActiveSum(partialScore);
        const float nextMaximum = max(runningMaximum, score);
        const float previousScale = exp(runningMaximum - nextMaximum);
        const float scoreScale = exp(score - nextMaximum);
        runningSum = runningSum * previousScale + scoreScale;
        for (uint slot = 0; slot < 16; ++slot) {
            const uint depth = lane + slot * laneCount;
            if (queryIsValid && depth < pc.valueDepth) {
                accumulated[slot] = accumulated[slot] * previousScale + scoreScale * sharedValue[depth];
            }
        }
        runningMaximum = nextMaximum;
        GroupMemoryBarrierWithGroupSync();
    }

    if (!queryIsValid) return;
    const uint outputOffset = (batch * pc.queryCount + queryIndex) * pc.valueDepth;
    for (uint slot = 0; slot < 16; ++slot) {
        const uint depth = lane + slot * laneCount;
        if (depth < pc.valueDepth) outputData[outputOffset + depth] = accumulated[slot] / runningSum;
    }
}
