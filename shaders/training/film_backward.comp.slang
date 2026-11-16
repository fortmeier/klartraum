// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// The backward pass of FiLM: workgroup (n, c) writes dx = dy * scale for its plane and reduces dscale = sum dy * x
// and dshift = sum dy over the plane's pixels in a fixed order.

struct PushConstants {
    uint batch;
    uint channels;
    uint pixels;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> x;
[[vk::binding(1, 0)]] StructuredBuffer<float> scale;
[[vk::binding(2, 0)]] StructuredBuffer<float> outputGradient;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> inputGradient;
[[vk::binding(4, 0)]] RWStructuredBuffer<float> scaleGradient;
[[vk::binding(5, 0)]] RWStructuredBuffer<float> shiftGradient;

groupshared float partialScale[256];
groupshared float partialShift[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group: SV_GroupID, uint3 local: SV_GroupThreadID) {
    const uint plane = group.x;
    const float s = scale[plane];
    float sumScale = 0.0, sumShift = 0.0;
    for (uint p = local.x; p < pc.pixels; p += 256u) {
        const uint i = plane * pc.pixels + p;
        const float g = outputGradient[i];
        inputGradient[i] = g * s;
        sumScale += g * x[i];
        sumShift += g;
    }
    partialScale[local.x] = sumScale;
    partialShift[local.x] = sumShift;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (local.x < stride) {
            partialScale[local.x] += partialScale[local.x + stride];
            partialShift[local.x] += partialShift[local.x + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (local.x == 0u) {
        scaleGradient[plane] = partialScale[0];
        shiftGradient[plane] = partialShift[0];
    }
}
