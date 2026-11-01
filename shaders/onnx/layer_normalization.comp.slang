// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Layer normalization over the last axis, then scale and bias per element.
// One subgroup per row (eight per workgroup): the lanes read strided values,
// so neighbouring lanes read neighbouring addresses, and the subgroup adds up
// the mean and then the centred squares.
import subgroup;

struct PushConstants {
    uint outerCount;
    uint axisSize;
    float epsilon;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> inputData;
[[vk::binding(1, 0)]] StructuredBuffer<float> scale;
[[vk::binding(2, 0)]] StructuredBuffer<float> bias;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> outputData;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 groupId: SV_GroupID) {
    const uint row = groupId.x * subgroupCount() + subgroupId();
    if (row >= pc.outerCount) return;
    const uint lane = WaveGetLaneIndex(), lanes = WaveGetLaneCount();
    const uint offset = row * pc.axisSize;
    float sum = 0.0;
    for (uint i = lane; i < pc.axisSize; i += lanes) sum += inputData[offset + i];
    const float mean = WaveActiveSum(sum) / float(pc.axisSize);
    float variance = 0.0;
    for (uint i = lane; i < pc.axisSize; i += lanes) {
        const float delta = inputData[offset + i] - mean;
        variance += delta * delta;
    }
    const float inverseDeviation = rsqrt(WaveActiveSum(variance) / float(pc.axisSize) + pc.epsilon);
    for (uint i = lane; i < pc.axisSize; i += lanes) {
        outputData[offset + i] = (inputData[offset + i] - mean) * inverseDeviation * scale[i] + bias[i];
    }
}
