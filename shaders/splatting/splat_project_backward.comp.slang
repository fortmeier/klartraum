// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Gradients of the projection of splat_project.comp.slang with respect to each Gaussian's parameters, from the
// gradients of its screen mean, conic, opacity and colour (splat_render_backward.comp.slang), by bwd_diff of the
// functions in splat_common.slang. Writes every Gaussian's parameter gradients (zero if culled) and adds the length
// of its screen-mean gradient and one to its densification statistics when it was visible.
import splat_common;

struct PushConstants {
    uint count;
    uint width;
    uint height;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float4> means;
[[vk::binding(1, 0)]] StructuredBuffer<float4> logScales;
[[vk::binding(2, 0)]] StructuredBuffer<float4> rotations;
[[vk::binding(3, 0)]] StructuredBuffer<float> opacities;
[[vk::binding(4, 0)]] StructuredBuffer<float4x4> cameras;
[[vk::binding(5, 0)]] StructuredBuffer<uint> viewIndex;
[[vk::binding(6, 0)]] StructuredBuffer<float4> projected;      // three per Gaussian
[[vk::binding(7, 0)]] StructuredBuffer<float> splatGradients;  // twelve per Gaussian
[[vk::binding(8, 0)]] RWStructuredBuffer<float4> meanGradients;
[[vk::binding(9, 0)]] RWStructuredBuffer<float4> logScaleGradients;
[[vk::binding(10, 0)]] RWStructuredBuffer<float4> rotationGradients;
[[vk::binding(11, 0)]] RWStructuredBuffer<float4> colorGradients;
[[vk::binding(12, 0)]] RWStructuredBuffer<float> opacityGradients;
[[vk::binding(13, 0)]] RWStructuredBuffer<float2> statistics; // per Gaussian: sum of |dL/dmean|, views seen

[shader("compute")]
[numthreads(128, 1, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    const uint g = globalId.x;
    if (g >= pc.count) return;
    if (projected[3 * g].w <= 0.0) { // culled
        meanGradients[g] = 0.0;
        logScaleGradients[g] = 0.0;
        rotationGradients[g] = 0.0;
        colorGradients[g] = 0.0;
        opacityGradients[g] = 0.0;
        return;
    }
    const float4x4 camera = cameras[viewIndex[0]];
    const float2 size = float2(pc.width, pc.height);
    const uint base = 12u * g;
    const float2 meanGradient = float2(splatGradients[base], splatGradients[base + 1u]);
    const float3 conicGradient = float3(splatGradients[base + 2u], splatGradients[base + 3u], splatGradients[base + 4u]);
    var position = diffPair(means[g].xyz);
    bwd_diff(projectMean)(position, camera, size, float3(meanGradient, 0.0));
    const float3 positionGradient = position.d;
    const float3 covariance = projectCovariance(means[g].xyz, logScales[g].xyz, rotations[g], camera, size);
    var covariancePair = diffPair(covariance);
    bwd_diff(conicOf)(covariancePair, conicGradient);
    var position2 = diffPair(means[g].xyz);
    var logScale = diffPair(logScales[g].xyz);
    var rotation = diffPair(rotations[g]);
    bwd_diff(projectCovariance)(position2, logScale, rotation, camera, size, covariancePair.d);
    var opacity = diffPair(opacities[g]);
    bwd_diff(sigmoid)(opacity, splatGradients[base + 5u]);
    meanGradients[g] = float4(positionGradient + position2.d, 0.0);
    logScaleGradients[g] = float4(logScale.d, 0.0);
    rotationGradients[g] = rotation.d;
    colorGradients[g] = float4(splatGradients[base + 6u], splatGradients[base + 7u], splatGradients[base + 8u], 0.0);
    opacityGradients[g] = opacity.d;
    statistics[g] += float2(length(meanGradient), 1.0);
}
