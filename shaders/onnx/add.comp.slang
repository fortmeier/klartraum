// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import broadcast;

struct PushConstants {
    uint elementCount;
    uint rank;
    uint lhsDims[4];
    uint rhsDims[4];
    uint outputDims[4];
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> lhs;
[[vk::binding(1, 0)]] StructuredBuffer<float> rhs;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> outputData;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    uint index = globalId.x;
    if (index < pc.elementCount) {
        outputData[index] = lhs[broadcastIndex(index, pc.lhsDims, pc.outputDims)]
                            + rhs[broadcastIndex(index, pc.rhsDims, pc.outputDims)];
    }
}
