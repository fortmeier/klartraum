// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Carries per-pixel colour gradients (e.g. from antialias_backward) back to texels and vertex positions (see
// backpropagateColor() in differentiable_shading.slang).
import differentiable_shading;

[[vk::binding(7, 0)]] StructuredBuffer<float4> colorGradients;    // rgb used; per pixel
[[vk::binding(8, 0)]] RWStructuredBuffer<float> vertexGradients;  // four per vertex
[[vk::binding(9, 0)]] RWStructuredBuffer<float> textureGradients; // four per texel

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    PixelSurface s;
    if (globalId.x >= pc.width || globalId.y >= pc.height || !pixelSurface(globalId.x, globalId.y, s)) return;
    const float3 gradient = colorGradients[globalId.y * pc.width + globalId.x].rgb;
    if (any(gradient != 0.0)) backpropagateColor(s, gradient, vertexGradients, textureGradients);
}
