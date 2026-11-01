// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

struct PushConstants {
    uint batch;
    uint channels;
    uint inputHeight;
    uint inputWidth;
    uint outputHeight;
    uint outputWidth;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> inputData;
[[vk::binding(1, 0)]] StructuredBuffer<float> scales;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> outputData;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    uint index = globalId.x;
    uint total = pc.batch * pc.channels * pc.outputHeight * pc.outputWidth;
    if (index >= total) return;
    uint x = index % pc.outputWidth;
    uint value = index / pc.outputWidth;
    uint y = value % pc.outputHeight;
    value /= pc.outputHeight;
    uint channel = value % pc.channels;
    uint batch = value / pc.channels;
    uint inputX = min(x * pc.inputWidth / pc.outputWidth, pc.inputWidth - 1);
    uint inputY = min(y * pc.inputHeight / pc.outputHeight, pc.inputHeight - 1);
    uint inputIndex = ((batch * pc.channels + channel) * pc.inputHeight + inputY) * pc.inputWidth + inputX;
    outputData[index] = inputData[inputIndex];
}
