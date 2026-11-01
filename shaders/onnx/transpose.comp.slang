// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import transpose_common;

[[vk::push_constant]] ConstantBuffer<TransposePushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> inputData;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> outputData;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    uint outputIndex = globalId.x;
    uint total = 1;
    for (uint axis = 0; axis < pc.dims; ++axis) total *= pc.dimOutput[axis];
    if (outputIndex >= total) return;
    uint outputCoordinates[8];
    uint remaining = outputIndex;
    for (int axis = int(pc.dims) - 1; axis >= 0; --axis) {
        outputCoordinates[axis] = remaining % pc.dimOutput[axis];
        remaining /= pc.dimOutput[axis];
    }
    uint inputCoordinates[8];
    for (uint axis = 0; axis < pc.dims; ++axis) inputCoordinates[pc.dimPerm[axis]] = outputCoordinates[axis];
    uint inputIndex = 0;
    uint stride = 1;
    for (int axis = int(pc.dims) - 1; axis >= 0; --axis) {
        inputIndex += inputCoordinates[axis] * stride;
        stride *= pc.dimInput[axis];
    }
    outputData[outputIndex] = inputData[inputIndex];
}
