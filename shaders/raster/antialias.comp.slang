// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Antialiases the silhouettes of shaded, premultiplied colours (see antialias.slang): each pixel blends towards
// every neighbour whose surface covers part of it.
import antialias;
import differentiable_shading;

[[vk::binding(7, 0)]] StructuredBuffer<float4> colors;       // premultiplied rgb, coverage; per pixel
[[vk::binding(8, 0)]] RWStructuredBuffer<float4> antialiased; // per pixel

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    if (globalId.x >= pc.width || globalId.y >= pc.height) return;
    const uint2 self = globalId.xy;
    const float4 own = colors[self.y * pc.width + self.x];
    float4 result = own;
    for (uint n = 0; n < 4; ++n) {
        const int2 other = int2(self) + NEIGHBOURS[n];
        if (any(other < 0) || other.x >= int(pc.width) || other.y >= int(pc.height)) continue;
        const float w = blendWeight(findSilhouette(self, uint2(other)));
        if (w > 0.0) result += w * (colors[other.y * pc.width + other.x] - own);
    }
    antialiased[self.y * pc.width + self.x] = result;
}
