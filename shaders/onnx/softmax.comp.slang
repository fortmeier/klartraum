// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

struct PushConstants {
    uint outerCount;
    uint axisSize;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> inputData;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> outputData;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    uint row = globalId.x;
    if (row >= pc.outerCount) return;
    uint offset = row * pc.axisSize;
    float maximum = inputData[offset];
    for (uint i = 1; i < pc.axisSize; ++i) maximum = max(maximum, inputData[offset + i]);
    float sum = 0.0;
    for (uint i = 0; i < pc.axisSize; ++i) sum += exp(inputData[offset + i] - maximum);
    for (uint i = 0; i < pc.axisSize; ++i) outputData[offset + i] = exp(inputData[offset + i] - maximum) / sum;
}
