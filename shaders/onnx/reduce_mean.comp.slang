// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// One invocation per output element; neighbouring invocations read
// neighbouring inner elements, so reductions over leading axes stay coalesced.

struct PushConstants {
    uint outerCount;
    uint axisSize;
    uint innerSize;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> inputData;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> outputData;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    uint index = globalId.x;
    if (index >= pc.outerCount * pc.innerSize) return;
    uint outer = index / pc.innerSize;
    uint inner = index - outer * pc.innerSize;
    uint offset = outer * pc.axisSize * pc.innerSize + inner;
    float sum = 0.0;
    for (uint i = 0; i < pc.axisSize; ++i) sum += inputData[offset + i * pc.innerSize];
    outputData[index] = sum / float(pc.axisSize);
}
