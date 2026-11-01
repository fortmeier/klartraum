// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

struct PushConstants {
    uint outerCount;
    uint lhsAxisSize;
    uint rhsAxisSize;
    uint innerSize;
    uint elementCount;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> lhs;
[[vk::binding(1, 0)]] StructuredBuffer<float> rhs;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> outputData;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    uint index = globalId.x;
    if (index >= pc.elementCount) return;
    uint outputBlock = (pc.lhsAxisSize + pc.rhsAxisSize) * pc.innerSize;
    uint outer = index / outputBlock;
    uint within = index % outputBlock;
    uint lhsBlock = pc.lhsAxisSize * pc.innerSize;
    outputData[index] = within < lhsBlock ? lhs[outer * lhsBlock + within]
                                          : rhs[outer * pc.rhsAxisSize * pc.innerSize + within - lhsBlock];
}
