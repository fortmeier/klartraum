// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Adds the specular term (specular.slang) to every covered pixel's shaded colour.
import differentiable_shading;
import specular;

[[vk::binding(7, 0)]] RWStructuredBuffer<float4> colors; // rgb, coverage; per pixel, added to

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    if (globalId.x >= pc.width || globalId.y >= pc.height) return;
    PixelSurface surface;
    if (!pixelSurface(globalId.x, globalId.y, surface)) return;
    const SpecularSurface s = specularSurface(surface);
    const float4 m = materialAt(s);
    const float grey = specularOf(m.xyz, m.w, lights[0], lights[1], lights[2], lights[3], lights[LIGHTS].x, s.normal,
                                  s.toEye);
    const uint pixel = globalId.y * pc.width + globalId.x;
    colors[pixel] = colors[pixel] + float4(grey, grey, grey, 0.0);
}
