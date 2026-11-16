// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Gradients of the antialiasing (see antialias.slang). For pixel p with blends towards neighbours q_k,
//   out_p = c_p + sum_k w_k (c_q_k - c_p),
// so c_p gets (1 - sum_k w_k) times the output gradient, each c_q_k gets w_k times it (added atomically, as several
// pixels may blend towards one), and w_k = +-(t_k - 0.5) carries dot(gradient, c_q_k - c_p) to the silhouette edge's
// vertices through bwd_diff of crossing().
import antialias;
import differentiable_shading;

[[vk::binding(7, 0)]] StructuredBuffer<float4> colors;              // premultiplied, per pixel
[[vk::binding(8, 0)]] StructuredBuffer<float4> antialiasedGradients; // per pixel
[[vk::binding(9, 0)]] RWStructuredBuffer<float> colorGradients;     // four per pixel
[[vk::binding(10, 0)]] RWStructuredBuffer<float> vertexGradients;   // four per vertex

void addColorGradient(uint pixel, float4 gradient) {
    for (uint c = 0; c < 4; ++c) InterlockedAdd(colorGradients[4 * pixel + c], gradient[c]);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    if (globalId.x >= pc.width || globalId.y >= pc.height) return;
    const uint2 self = globalId.xy;
    const uint pixel = self.y * pc.width + self.x;
    const float4 gradient = antialiasedGradients[pixel];
    const float4 own = colors[pixel];
    float ownWeight = 1.0;
    const float2 size = float2(pc.width, pc.height);
    for (uint n = 0; n < 4; ++n) {
        const int2 other = int2(self) + NEIGHBOURS[n];
        if (any(other < 0) || other.x >= int(pc.width) || other.y >= int(pc.height)) continue;
        const Silhouette s = findSilhouette(self, uint2(other));
        const float w = blendWeight(s);
        if (w <= 0.0) continue;
        const uint otherPixel = uint(other.y) * pc.width + uint(other.x);
        ownWeight -= w;
        addColorGradient(otherPixel, w * gradient);
        // w = 0.5 - t if this pixel holds the front triangle, else t - 0.5.
        const float tGradient = dot(gradient, colors[otherPixel] - own) * (s.frontIsSelf ? -1.0 : 1.0);
        var a = diffPair(s.a);
        var b = diffPair(s.b);
        bwd_diff(crossing)(a, b, s.camera, s.frontPixel, s.otherPixel, size, tGradient);
        for (uint c = 0; c < 3; ++c) {
            InterlockedAdd(vertexGradients[4 * s.vertexA + c], a.d[c]);
            InterlockedAdd(vertexGradients[4 * s.vertexB + c], b.d[c]);
        }
    }
    addColorGradient(pixel, ownWeight * gradient);
}
