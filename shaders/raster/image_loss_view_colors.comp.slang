// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// The image loss of image_loss.comp.slang with a colour correction per view, for photographs whose exposure and white
// balance differ: the rendered premultiplied colour c (coverage a) is compared as gain x c + offset x a, with gain
// and offset (rgb) of the drawn view. A pull of the corrections towards identity,
//   viewColorWeight x (|gain - 1|^2 + |offset|^2),
// keeps the texture from trading brightness with them. Writes each pixel's gradient, adds the loss and the gradients
// of the drawn view's gain and offset.
import subgroup;

struct PushConstants {
    uint width;
    uint height;
    float maskWeight;
    float viewColorWeight;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float4> colors;              // premultiplied rgb, coverage; per pixel
[[vk::binding(1, 0)]] StructuredBuffer<float4> targets;             // rgb, mask; per view and pixel
[[vk::binding(2, 0)]] StructuredBuffer<uint> viewIndex;
[[vk::binding(3, 0)]] RWStructuredBuffer<float4> gradients;         // per pixel
[[vk::binding(4, 0)]] RWStructuredBuffer<float> loss;               // one element, added to
[[vk::binding(5, 0)]] StructuredBuffer<float4> viewColors;          // gain, offset (rgb) per view
[[vk::binding(6, 0)]] RWStructuredBuffer<float> viewColorGradients; // eight per view, added to

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    const uint pixelCount = pc.width * pc.height;
    const uint pixel = globalId.x;
    const uint view = viewIndex[0];
    const float3 gain = viewColors[2 * view].rgb;
    const float3 offset = viewColors[2 * view + 1].rgb;
    float pixelLoss = 0.0;
    float3 gainGradient = 0.0;
    float3 offsetGradient = 0.0;
    if (pixel < pixelCount) {
        const float4 color = colors[pixel];
        const float4 target = targets[view * pixelCount + pixel];
        const float mask = target.a;
        const float3 difference = gain * color.rgb + offset * color.a - target.rgb;
        const float coverageDifference = color.a - mask;
        const float scale = 1.0 / float(pixelCount);
        pixelLoss = (mask * dot(difference, difference) + pc.maskWeight * coverageDifference * coverageDifference) *
                    scale;
        const float3 colorGradient = 2.0 * mask * difference * scale;
        gradients[pixel] = float4(colorGradient * gain,
                                  2.0 * pc.maskWeight * coverageDifference * scale + dot(colorGradient, offset));
        gainGradient = colorGradient * color.rgb;
        offsetGradient = colorGradient * color.a;
    }
    if (globalId.x == 0) {
        pixelLoss += pc.viewColorWeight * (dot(gain - 1.0, gain - 1.0) + dot(offset, offset));
        gainGradient += 2.0 * pc.viewColorWeight * (gain - 1.0);
        offsetGradient += 2.0 * pc.viewColorWeight * offset;
    }
    const float groupLoss = WaveActiveSum(pixelLoss);
    const float3 groupGain = WaveActiveSum(gainGradient);
    const float3 groupOffset = WaveActiveSum(offsetGradient);
    if (WaveIsFirstLane()) {
        InterlockedAdd(loss[0], groupLoss);
        for (uint c = 0; c < 3; ++c) {
            InterlockedAdd(viewColorGradients[8 * view + c], groupGain[c]);
            InterlockedAdd(viewColorGradients[8 * view + 4 + c], groupOffset[c]);
        }
    }
}
