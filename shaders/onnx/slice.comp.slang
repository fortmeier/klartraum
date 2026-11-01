// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import slice_common;

[[vk::push_constant]] ConstantBuffer<SlicePushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> inputData;
[[vk::binding(1, 0)]] StructuredBuffer<uint> starts;
[[vk::binding(2, 0)]] StructuredBuffer<uint> ends;
[[vk::binding(3, 0)]] StructuredBuffer<uint> axes;
[[vk::binding(4, 0)]] RWStructuredBuffer<float> outputData;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    uint outputIndex = globalId.x;
    if (outputIndex >= pc.elementCount) return;
    outputData[outputIndex] = inputData[sliceInputIndex(outputIndex, pc)];
}
