// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Gradients of the specular term (specular.slang) from the gradients of the shaded colours: the term adds the same
// grey to r, g and b, so its gradient is their sum. bwd_diff of specularOf() gives the gradients of the material at
// the pixel, spread to its four texels by their bilinear weights, and of the lights, which a subgroup sums before one
// lane adds them.
import differentiable_shading;
import specular;

[[vk::binding(7, 0)]] StructuredBuffer<float4> colorGradients;     // per pixel, rgb used
[[vk::binding(12, 0)]] RWStructuredBuffer<float> materialGradients; // four per texel, added to
[[vk::binding(13, 0)]] RWStructuredBuffer<float> lightGradients;    // four per light and four for the exponent

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    float4 lightGradient[LIGHTS + 1];
    for (uint k = 0; k <= LIGHTS; ++k) lightGradient[k] = 0.0;
    PixelSurface surface;
    const bool inside = globalId.x < pc.width && globalId.y < pc.height;
    if (inside && pixelSurface(globalId.x, globalId.y, surface)) {
        const SpecularSurface s = specularSurface(surface);
        const float4 m = materialAt(s);
        const float3 g = colorGradients[globalId.y * pc.width + globalId.x].rgb;
        var offset = diffPair(m.xyz);
        var amplitude = diffPair(m.w);
        var l0 = diffPair(lights[0]);
        var l1 = diffPair(lights[1]);
        var l2 = diffPair(lights[2]);
        var l3 = diffPair(lights[3]);
        var exponent = diffPair(lights[LIGHTS].x);
        bwd_diff(specularOf)(offset, amplitude, l0, l1, l2, l3, exponent, s.normal, s.toEye, g.r + g.g + g.b);
        const float4 materialGradient = float4(offset.d, amplitude.d);
        const float4 w = bilinearWeights(s.taps.fraction);
        for (uint k = 0; k < 4; ++k) {
            for (uint c = 0; c < 4; ++c)
                InterlockedAdd(materialGradients[4 * s.taps.index[k] + c], w[k] * materialGradient[c]);
        }
        lightGradient[0] = l0.d;
        lightGradient[1] = l1.d;
        lightGradient[2] = l2.d;
        lightGradient[3] = l3.d;
        lightGradient[LIGHTS] = float4(exponent.d, 0.0, 0.0, 0.0);
    }
    for (uint k = 0; k <= LIGHTS; ++k) {
        const float4 sum = WaveActiveSum(lightGradient[k]);
        if (WaveIsFirstLane() && any(sum != 0.0)) {
            for (uint c = 0; c < 4; ++c) InterlockedAdd(lightGradients[4 * k + c], sum[c]);
        }
    }
}
