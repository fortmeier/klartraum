// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

struct PushConstants {
    uint elementCount;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> inputData;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> outputData;

float errorFunction(float value) {
    float magnitude = abs(value);
    float t = 1.0 / (1.0 + 0.3275911 * magnitude);
    float polynomial = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                       - 0.284496736) * t + 0.254829592) * t;
    float result = 1.0 - polynomial * exp(-magnitude * magnitude);
    return value < 0.0 ? -result : result;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    uint index = globalId.x;
    if (index < pc.elementCount) outputData[index] = errorFunction(inputData[index]);
}
