// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// The scale and shift gradients of ChannelNorm: per channel the sums of dy * x^ and of dy over the batch and the
// pixels. One workgroup per channel, reducing in a fixed order.

struct PushConstants {
    uint batch;
    uint channels;
    uint pixels;
    float epsilon;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> x;
[[vk::binding(1, 0)]] StructuredBuffer<float> stats;
[[vk::binding(2, 0)]] StructuredBuffer<float> outputGradient;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> weightGradient;
[[vk::binding(4, 0)]] RWStructuredBuffer<float> biasGradient;

groupshared float partialWeight[256];
groupshared float partialBias[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group: SV_GroupID, uint3 local: SV_GroupThreadID) {
    const uint c = group.x;
    float sumWeight = 0.0, sumBias = 0.0;
    for (uint k = local.x; k < pc.batch * pc.pixels; k += 256u) {
        const uint n = k / pc.pixels, p = k % pc.pixels;
        const uint i = (n * pc.channels + c) * pc.pixels + p;
        const float normalized = (x[i] - stats[(n * 2u) * pc.pixels + p]) * stats[(n * 2u + 1u) * pc.pixels + p];
        sumWeight += outputGradient[i] * normalized;
        sumBias += outputGradient[i];
    }
    partialWeight[local.x] = sumWeight;
    partialBias[local.x] = sumBias;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (local.x < stride) {
            partialWeight[local.x] += partialWeight[local.x + stride];
            partialBias[local.x] += partialBias[local.x + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (local.x == 0u) {
        weightGradient[c] = partialWeight[0];
        biasGradient[c] = partialBias[0];
    }
}
