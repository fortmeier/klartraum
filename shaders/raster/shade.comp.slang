// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Shades a visibility buffer (see differentiable_shading.slang): per pixel the textured colour and a coverage of 1,
// or zero for background.
import differentiable_shading;

[[vk::binding(7, 0)]] RWStructuredBuffer<float4> colors; // rgb, coverage; per pixel

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    if (globalId.x >= pc.width || globalId.y >= pc.height) return;
    PixelSurface surface;
    const uint pixel = globalId.y * pc.width + globalId.x;
    colors[pixel] = pixelSurface(globalId.x, globalId.y, surface) ? float4(shade(surface), 1.0) : float4(0.0);
}
