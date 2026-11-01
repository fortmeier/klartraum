// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

struct PushConstants {
    uint batch;
    uint channels;
    uint spatialSize;
    float epsilon;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> inputData;
[[vk::binding(1, 0)]] StructuredBuffer<float> scale;
[[vk::binding(2, 0)]] StructuredBuffer<float> bias;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> outputData;

// Per-lane partial sums of the workgroup's tree reduction.
groupshared float partial[64];

// Sums `value` over the 64 lanes of the workgroup.
float workgroupSum(uint lane, float value) {
    partial[lane] = value;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 32; stride > 0; stride /= 2) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        GroupMemoryBarrierWithGroupSync();
    }
    return partial[0];
}

// One workgroup per (n, c) slice.
[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 localId: SV_GroupThreadID, uint3 groupId: SV_GroupID) {
    const uint lane = localId.x;
    const uint nc = groupId.x;
    if (nc >= pc.batch * pc.channels) return;
    const uint offset = nc * pc.spatialSize;

    float sum = 0.0;
    for (uint i = lane; i < pc.spatialSize; i += 64) sum += inputData[offset + i];
    const float mean = workgroupSum(lane, sum) / float(pc.spatialSize);

    float sumSquared = 0.0;
    for (uint i = lane; i < pc.spatialSize; i += 64) {
        const float centered = inputData[offset + i] - mean;
        sumSquared += centered * centered;
    }
    const float variance = workgroupSum(lane, sumSquared) / float(pc.spatialSize);
    float inverseStd = rsqrt(variance + pc.epsilon);
    uint channel = nc % pc.channels;
    for (uint i = lane; i < pc.spatialSize; i += 64) {
        outputData[offset + i] = (inputData[offset + i] - mean) * inverseStd * scale[channel] + bias[channel];
    }
}
