// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// The depth of a visibility buffer's surface (see differentiable_shading.slang): per pixel the clip w of the nearest
// triangle at the pixel centre, infinity for background. Gaussians blended in front of the mesh stop there.
import differentiable_shading;

[[vk::binding(7, 0)]] RWStructuredBuffer<float> depths; // per pixel

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    if (globalId.x >= pc.width || globalId.y >= pc.height) return;
    PixelSurface surface;
    const uint pixel = globalId.y * pc.width + globalId.x;
    depths[pixel] = pixelSurface(globalId.x, globalId.y, surface) ? surfaceDepth(surface) : 1.0 / 0.0;
}
