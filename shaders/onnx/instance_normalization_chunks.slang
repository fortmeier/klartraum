// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Instance normalization of large slices in two passes over chunks, so that
// many workgroups share a slice (N x C slices of spatialSize values each):
//  1. instance_normalization_stats: the workgroup of (chunk, slice) stores its
//     chunk's mean and sum of squared deviations (M2), centred within the
//     chunk;
//  2. instance_normalization_apply: on the same grid, every workgroup merges
//     its slice's chunk statistics in chunk order (Chan et al.) and
//     normalizes its chunk.
// The merge is exact in its order and stable, and the result does not depend
// on scheduling.
module instance_normalization_chunks;

import subgroup;

public struct InstanceNormalizationPushConstants {
    public uint batch;
    public uint channels;
    public uint spatialSize;
    public float epsilon;
    public uint chunkSize;
    public uint chunks;
};
[[vk::push_constant]] public ConstantBuffer<InstanceNormalizationPushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> inputData;
[[vk::binding(1, 0)]] StructuredBuffer<float> scale;
[[vk::binding(2, 0)]] StructuredBuffer<float> bias;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> outputData;
// Per (slice, chunk): mean and M2.
[[vk::binding(4, 0)]] RWStructuredBuffer<float2> chunkStatistics;

// Each subgroup's partial sum of workgroupSum().
groupshared float partial[8];

// Sums `value` over the 256 invocations of the workgroup (all must call it).
float workgroupSum(float value) {
    const float subgroupTotal = WaveActiveSum(value);
    if (WaveIsFirstLane()) partial[subgroupId()] = subgroupTotal;
    GroupMemoryBarrierWithGroupSync();
    float total = 0.0;
    for (uint s = 0; s < subgroupCount(); ++s) total += partial[s];
    GroupMemoryBarrierWithGroupSync();
    return total;
}

// Runs in a 256-wide workgroup of the (chunks, N x C) grid.
public void computeChunkStatistics(uint localIndex, uint3 groupId) {
    const uint slice = groupId.y, chunk = groupId.x;
    const uint begin = chunk * pc.chunkSize, end = min(pc.spatialSize, begin + pc.chunkSize);
    const uint offset = slice * pc.spatialSize;
    float sum = 0.0;
    for (uint i = begin + localIndex; i < end; i += 256) sum += inputData[offset + i];
    const float mean = workgroupSum(sum) / float(end - begin);
    float squares = 0.0;
    for (uint i = begin + localIndex; i < end; i += 256) {
        const float centered = inputData[offset + i] - mean;
        squares += centered * centered;
    }
    const float m2 = workgroupSum(squares);
    if (localIndex == 0) chunkStatistics[slice * pc.chunks + chunk] = float2(mean, m2);
}

// Runs in a 256-wide workgroup of the (chunks, N x C) grid.
public void applyNormalization(uint localIndex, uint3 groupId) {
    const uint slice = groupId.y, chunk = groupId.x;
    float count = 0.0, mean = 0.0, m2 = 0.0;
    for (uint c = 0; c < pc.chunks; ++c) {
        const float2 statistics = chunkStatistics[slice * pc.chunks + c];
        const float chunkCount = float(min(pc.spatialSize, (c + 1) * pc.chunkSize) - c * pc.chunkSize);
        const float total = count + chunkCount;
        const float delta = statistics.x - mean;
        mean += delta * (chunkCount / total);
        m2 += statistics.y + delta * delta * (count * chunkCount / total);
        count = total;
    }
    const float inverseStd = rsqrt(m2 / float(pc.spatialSize) + pc.epsilon);
    const uint channel = slice % pc.channels;
    const float factor = inverseStd * scale[channel], shift = bias[channel];
    const uint begin = chunk * pc.chunkSize, end = min(pc.spatialSize, begin + pc.chunkSize);
    const uint offset = slice * pc.spatialSize;
    for (uint i = begin + localIndex; i < end; i += 256) {
        outputData[offset + i] = (inputData[offset + i] - mean) * factor + shift;
    }
}
