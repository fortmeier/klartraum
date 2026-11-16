// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// The image loss of antialiased, premultiplied colours against a masked target image:
//   loss = sum over pixels of (m |colour - target colour|^2 + maskWeight (coverage - m)^2) / (width x height),
// with m the target mask. The colour term counts inside the mask only, where premultiplied colours dim when the
// mesh leaves part of the pixel uncovered; outside it, the coverage term alone pulls the silhouette in. Writes each
// pixel's gradient and adds the loss.
import subgroup;

struct PushConstants {
    uint width;
    uint height;
    float maskWeight;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float4> colors;      // premultiplied rgb, coverage; per pixel
[[vk::binding(1, 0)]] StructuredBuffer<float4> targets;     // rgb, mask; per view and pixel
[[vk::binding(2, 0)]] StructuredBuffer<uint> viewIndex;
[[vk::binding(3, 0)]] RWStructuredBuffer<float4> gradients; // per pixel
[[vk::binding(4, 0)]] RWStructuredBuffer<float> loss;       // one element

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    const uint pixelCount = pc.width * pc.height;
    const uint pixel = globalId.x;
    float pixelLoss = 0.0;
    if (pixel < pixelCount) {
        const float4 color = colors[pixel];
        const float4 target = targets[viewIndex[0] * pixelCount + pixel];
        const float mask = target.a;
        const float3 difference = color.rgb - target.rgb;
        const float coverageDifference = color.a - mask;
        pixelLoss = (mask * dot(difference, difference) + pc.maskWeight * coverageDifference * coverageDifference) /
                    float(pixelCount);
        gradients[pixel] = float4(2.0 * mask * difference, 2.0 * pc.maskWeight * coverageDifference) / float(pixelCount);
    }
    const float groupLoss = WaveActiveSum(pixelLoss);
    if (WaveIsFirstLane()) InterlockedAdd(loss[0], groupLoss);
}
