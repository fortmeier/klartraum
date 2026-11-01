// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

struct PushConstants {
    uint outerCount;
    uint axisSize;
    uint innerSize;
    uint indexCount;
    uint outputCount;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> data;
[[vk::binding(1, 0)]] StructuredBuffer<int64_t> indices;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> outputData;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    uint outputIndex = globalId.x;
    if (outputIndex >= pc.outputCount) return;
    uint outer = outputIndex / (pc.indexCount * pc.innerSize);
    uint remainder = outputIndex % (pc.indexCount * pc.innerSize);
    uint indexOffset = remainder / pc.innerSize;
    uint inner = remainder % pc.innerSize;
    int64_t selected = indices[indexOffset];
    if (selected < 0) selected += int64_t(pc.axisSize);
    if (selected < 0 || selected >= int64_t(pc.axisSize)) {
        outputData[outputIndex] = 0.0;
        return;
    }
    uint inputIndex = (outer * pc.axisSize + uint(selected)) * pc.innerSize + inner;
    outputData[outputIndex] = data[inputIndex];
}
