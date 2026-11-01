// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import slice_common;

[[vk::push_constant]] ConstantBuffer<SlicePushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<int64_t> inputData;
[[vk::binding(1, 0)]] StructuredBuffer<int64_t> starts;
[[vk::binding(2, 0)]] StructuredBuffer<int64_t> ends;
[[vk::binding(3, 0)]] StructuredBuffer<int64_t> axes;
[[vk::binding(4, 0)]] RWStructuredBuffer<int64_t> outputData;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    uint outputIndex = globalId.x;
    if (outputIndex >= pc.elementCount) return;
    outputData[outputIndex] = inputData[sliceInputIndex(outputIndex, pc)];
}
