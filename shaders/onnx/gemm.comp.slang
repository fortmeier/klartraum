// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

struct PushConstants {
    uint rows;
    uint columns;
    uint reduction;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> inputData;
[[vk::binding(1, 0)]] StructuredBuffer<float> weights;
[[vk::binding(2, 0)]] StructuredBuffer<float> bias;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> outputData;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    uint column = globalId.x;
    uint row = globalId.y;
    if (row >= pc.rows || column >= pc.columns) return;
    float value = bias[column];
    for (uint k = 0; k < pc.reduction; ++k) {
        value += inputData[row * pc.reduction + k] * weights[column * pc.reduction + k];
    }
    outputData[row * pc.columns + column] = value;
}
