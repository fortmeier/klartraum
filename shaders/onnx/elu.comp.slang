// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

struct PushConstants {
    uint elementCount;
    float alpha;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> inputData;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> outputData;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    uint index = globalId.x;
    if (index < pc.elementCount) {
        float x = inputData[index];
        outputData[index] = x > 0.0 ? x : pc.alpha * (exp(x) - 1.0);
    }
}
