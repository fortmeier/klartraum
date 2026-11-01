// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// RMS normalization over the last axis: x / sqrt(mean(x^2) + eps) * scale.
// One 32-wide subgroup per row: the lanes sum strided squares, the subgroup
// adds them up, and the lanes write the normalized row.
import subgroup;

struct PushConstants {
    uint rows;
    uint width;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> inputData;
[[vk::binding(1, 0)]] StructuredBuffer<float> eps;
[[vk::binding(2, 0)]] StructuredBuffer<float> scale;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> outputData;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 groupId: SV_GroupID) {
    const uint row = groupId.x * (256 / 32) + subgroupId();
    if (row >= pc.rows) return;
    const uint lane = WaveGetLaneIndex();
    const uint offset = row * pc.width;
    float sum = 0.0;
    for (uint i = lane; i < pc.width; i += WaveGetLaneCount()) {
        const float value = inputData[offset + i];
        sum += value * value;
    }
    const float inverse = rsqrt(WaveActiveSum(sum) / float(pc.width) + eps[0]);
    for (uint i = lane; i < pc.width; i += WaveGetLaneCount()) {
        outputData[offset + i] = inputData[offset + i] * inverse * scale[i];
    }
}
