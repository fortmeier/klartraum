// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

struct PushConstants {
    uint elementCount;
    uint rank;
    uint inputDims[4];
    uint outputDims[4];
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<int64_t> inputData;
// The target shape; the dimensions come in through the push constants.
[[vk::binding(1, 0)]] StructuredBuffer<int64_t> shapeData;
[[vk::binding(2, 0)]] RWStructuredBuffer<int64_t> outputData;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    uint index = globalId.x;
    if (index >= pc.elementCount) return;
    uint remaining = index;
    uint inputIndex = 0;
    uint inputStride = 1;
    for (int axis = 3; axis >= 0; --axis) {
        uint coordinate = remaining % pc.outputDims[axis];
        remaining /= pc.outputDims[axis];
        inputIndex += (pc.inputDims[axis] == 1 ? 0 : coordinate) * inputStride;
        inputStride *= pc.inputDims[axis];
    }
    outputData[index] = inputData[inputIndex];
}
