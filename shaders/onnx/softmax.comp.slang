// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Softmax over the last axis. One subgroup per row (eight per workgroup): the
// lanes read strided values, so neighbouring lanes read neighbouring
// addresses, and the subgroup combines the maximum and the sum.
import subgroup;

struct PushConstants {
    uint outerCount;
    uint axisSize;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> inputData;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> outputData;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 groupId: SV_GroupID) {
    const uint row = groupId.x * subgroupCount() + subgroupId();
    if (row >= pc.outerCount) return;
    const uint lane = WaveGetLaneIndex(), lanes = WaveGetLaneCount();
    const uint offset = row * pc.axisSize;
    float maximum = inputData[offset];
    for (uint i = lane; i < pc.axisSize; i += lanes) maximum = max(maximum, inputData[offset + i]);
    maximum = WaveActiveMax(maximum);
    float sum = 0.0;
    for (uint i = lane; i < pc.axisSize; i += lanes) sum += exp(inputData[offset + i] - maximum);
    const float inverseSum = 1.0 / WaveActiveSum(sum);
    for (uint i = lane; i < pc.axisSize; i += lanes) {
        outputData[offset + i] = exp(inputData[offset + i] - maximum) * inverseSum;
    }
}
