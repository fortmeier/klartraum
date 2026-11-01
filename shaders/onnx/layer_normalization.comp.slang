// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

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
[numthreads(64, 1, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    uint row = globalId.x;
    if (row >= pc.outerCount) return;
    uint offset = row * pc.axisSize;
    float mean = 0.0;
    for (uint i = 0; i < pc.axisSize; ++i) mean += inputData[offset + i];
    mean /= float(pc.axisSize);
    float variance = 0.0;
    for (uint i = 0; i < pc.axisSize; ++i) {
        float delta = inputData[offset + i] - mean;
        variance += delta * delta;
    }
    float inverseDeviation = rsqrt(variance / float(pc.axisSize) + pc.epsilon);
    for (uint i = 0; i < pc.axisSize; ++i) {
        outputData[offset + i] = (inputData[offset + i] - mean) * inverseDeviation * scale[i] + bias[i];
    }
}
