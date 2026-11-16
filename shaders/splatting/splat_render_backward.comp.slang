// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Gradients of the blending of splat_render.comp.slang with respect to each splat's screen mean, conic, opacity and
// colour. Each pixel walks its blended splats back to front (Kerbl et al. 2023): with T_i the transmittance before
// splat i and B_i the colour of the splats behind it (blended as if alone),
//   dL/dc_i = a_i T_i dL/dC,   dL/da_i = T_i dot(c_i - B_i, dL/dC) + dL/dCoverage T_final / (1 - a_i),
// and bwd_diff of splatAlpha() carries dL/da_i to mean, conic and opacity (not where the alpha is clamped). All
// pixels of a tile walk the same splat sequence, so a subgroup sums its pixels' gradients of a splat before adding
// them atomically (twelve floats per splat: mean xy, conic xx xy yy, opacity, rgb, 3 unused).
import splat_common;

struct PushConstants {
    uint width;
    uint height;
    uint tilesX;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float4> projected;       // three per Gaussian
[[vk::binding(1, 0)]] StructuredBuffer<uint> values;            // sorted instances: the Gaussian
[[vk::binding(2, 0)]] StructuredBuffer<uint2> ranges;           // per tile
[[vk::binding(3, 0)]] StructuredBuffer<float4> colors;          // per pixel: the forward output
[[vk::binding(4, 0)]] StructuredBuffer<uint> lastContributor;   // per pixel
[[vk::binding(5, 0)]] StructuredBuffer<float4> imageGradients;  // per pixel: dL/d(rgb, coverage)
[[vk::binding(6, 0)]] RWStructuredBuffer<float> splatGradients; // twelve per Gaussian, added to

static const uint BATCH = TILE * TILE;
groupshared float4 batchMean[BATCH];
groupshared float4 batchConic[BATCH];
groupshared float4 batchColor[BATCH];
groupshared uint batchGaussian[BATCH];

[shader("compute")]
[numthreads(TILE, TILE, 1)]
void main(uint3 groupId: SV_GroupID, uint3 localId: SV_GroupThreadID, uint localIndex: SV_GroupIndex) {
    const uint2 pixel = groupId.xy * TILE + localId.xy;
    const bool inside = pixel.x < pc.width && pixel.y < pc.height;
    const uint index = inside ? pixel.y * pc.width + pixel.x : 0u;
    const float2 centre = float2(pixel) + 0.5;
    const uint2 range = ranges[groupId.y * pc.tilesX + groupId.x];
    const uint last = inside ? range.x + lastContributor[index] : range.x; // this pixel blended [range.x, last)
    const float4 gradient = inside ? imageGradients[index] : float4(0.0);
    const float finalTransmittance = inside ? 1.0 - colors[index].a : 1.0;
    float transmittance = finalTransmittance;
    float3 behind = 0.0;
    const uint count = range.y - range.x;
    for (uint done = 0u; done < count; done += BATCH) {
        // Batches from the back: [start, end).
        const uint end = range.y - done;
        const uint start = end - min(BATCH, count - done);
        if (start + localIndex < end) {
            const uint g = values[start + localIndex];
            batchGaussian[localIndex] = g;
            batchMean[localIndex] = projected[3 * g];
            batchConic[localIndex] = projected[3 * g + 1];
            batchColor[localIndex] = projected[3 * g + 2];
        }
        GroupMemoryBarrierWithGroupSync();
        for (uint j = end - start; j-- > 0u;) {
            float2 meanGradient = 0.0;
            float3 conicGradient = 0.0;
            float opacityGradient = 0.0;
            float3 colorGradient = 0.0;
            const float2 mean = batchMean[j].xy;
            const float3 conic = batchConic[j].xyz;
            const float opacity = batchConic[j].w;
            const float3 color = batchColor[j].rgb;
            const float rawAlpha = splatAlpha(mean, conic, opacity, centre);
            const float alpha = min(rawAlpha, MAX_ALPHA);
            if (start + j < last && alpha >= MIN_ALPHA) {
                transmittance /= 1.0 - alpha;
                colorGradient = alpha * transmittance * gradient.rgb;
                const float alphaGradient = transmittance * dot(color - behind, gradient.rgb) +
                                            gradient.a * finalTransmittance / (1.0 - alpha);
                behind = alpha * color + (1.0 - alpha) * behind;
                if (rawAlpha < MAX_ALPHA) {
                    var meanPair = diffPair(mean);
                    var conicPair = diffPair(conic);
                    var opacityPair = diffPair(opacity);
                    bwd_diff(splatAlpha)(meanPair, conicPair, opacityPair, centre, alphaGradient);
                    meanGradient = meanPair.d;
                    conicGradient = conicPair.d;
                    opacityGradient = opacityPair.d;
                }
            }
            // Every lane of the subgroup holds the same splat j: sum, then one lane adds.
            const float2 meanSum = WaveActiveSum(meanGradient);
            const float3 conicSum = WaveActiveSum(conicGradient);
            const float opacitySum = WaveActiveSum(opacityGradient);
            const float3 colorSum = WaveActiveSum(colorGradient);
            const bool contributes =
                any(meanSum != 0.0) || any(conicSum != 0.0) || opacitySum != 0.0 || any(colorSum != 0.0);
            if (WaveIsFirstLane() && contributes) {
                const uint base = 12u * batchGaussian[j];
                InterlockedAdd(splatGradients[base], meanSum.x);
                InterlockedAdd(splatGradients[base + 1u], meanSum.y);
                InterlockedAdd(splatGradients[base + 2u], conicSum.x);
                InterlockedAdd(splatGradients[base + 3u], conicSum.y);
                InterlockedAdd(splatGradients[base + 4u], conicSum.z);
                InterlockedAdd(splatGradients[base + 5u], opacitySum);
                InterlockedAdd(splatGradients[base + 6u], colorSum.x);
                InterlockedAdd(splatGradients[base + 7u], colorSum.y);
                InterlockedAdd(splatGradients[base + 8u], colorSum.z);
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
}
