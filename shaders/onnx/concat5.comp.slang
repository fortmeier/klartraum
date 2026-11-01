// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import concat_n;

[[vk::push_constant]] ConstantBuffer<ConcatPushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> input0;
[[vk::binding(1, 0)]] StructuredBuffer<float> input1;
[[vk::binding(2, 0)]] StructuredBuffer<float> input2;
[[vk::binding(3, 0)]] StructuredBuffer<float> input3;
[[vk::binding(4, 0)]] StructuredBuffer<float> input4;
[[vk::binding(5, 0)]] RWStructuredBuffer<float> outputData;

struct Inputs : IConcatInputs {
    static const uint count = 5;
    static float load(uint input, uint index) {
        switch (input) {
        case 0: return input0[index];
        case 1: return input1[index];
        case 2: return input2[index];
        case 3: return input3[index];
        default: return input4[index];
        }
    }
};

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    if (let value = concatElement<Inputs>(globalId.x, pc)) outputData[globalId.x] = value;
}
