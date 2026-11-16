// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// The image loss of antialiased, premultiplied colours against a masked target image:
//   loss = sum over pixels of (m |colour - target colour|^2 + maskWeight (coverage - m)^2) / (width x height),
// with m the target mask. The colour term counts inside the mask only, where premultiplied colours dim when the
// mesh leaves part of the pixel uncovered; outside it, the coverage term alone pulls the silhouette in. Writes each
// pixel's gradient and adds the loss.
//
// With a highlight tolerance t > 0, a target pixel brighter than the colour by more (in luminance) is taken for a
// highlight that only this view sees, e.g. a specular reflection: its colour term is weighted by (t / excess)^2, a
// weight that the gradient takes as constant (iteratively reweighted least squares). A colour then follows what most
// views agree on rather than the brightest.
import subgroup;

struct PushConstants {
    uint width;
    uint height;
    float maskWeight;
    float highlightTolerance; // 0: off
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
        float weight = mask;
        const float excess = dot(target.rgb - color.rgb, float3(0.2126, 0.7152, 0.0722));
        if (pc.highlightTolerance > 0.0 && excess > pc.highlightTolerance) {
            const float ratio = pc.highlightTolerance / excess;
            weight *= ratio * ratio;
        }
        pixelLoss = (weight * dot(difference, difference) + pc.maskWeight * coverageDifference * coverageDifference) /
                    float(pixelCount);
        gradients[pixel] =
            float4(2.0 * weight * difference, 2.0 * pc.maskWeight * coverageDifference) / float(pixelCount);
    }
    const float groupLoss = WaveActiveSum(pixelLoss);
    if (WaveIsFirstLane()) InterlockedAdd(loss[0], groupLoss);
}
