// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Projects every Gaussian under the drawn view (splat_common.slang) and emits one sort instance per tile its
// three-sigma square touches: key (tile << DEPTH_BITS) | quantized depth, value the Gaussian. Instances beyond the
// capacity are dropped; the counter keeps counting, so overflow is visible.
import splat_common;

struct PushConstants {
    uint count;
    uint width;
    uint height;
    uint tilesX;
    uint tilesY;
    uint capacity;
    float depthScale; // quantization steps per unit of clip w
    float nearDepth;  // Gaussians with a smaller clip w are culled
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float4> means;      // xyz per Gaussian
[[vk::binding(1, 0)]] StructuredBuffer<float4> logScales;  // xyz per Gaussian
[[vk::binding(2, 0)]] StructuredBuffer<float4> rotations;  // quaternion (x, y, z, w) per Gaussian
[[vk::binding(3, 0)]] StructuredBuffer<float4> colors;     // rgb per Gaussian
[[vk::binding(4, 0)]] StructuredBuffer<float> opacities;   // logit per Gaussian
[[vk::binding(5, 0)]] StructuredBuffer<float4x4> cameras;  // clip from world per view
[[vk::binding(6, 0)]] StructuredBuffer<uint> viewIndex;    // the drawn view
// Three per Gaussian: (mean x, mean y, depth, radius; radius 0: culled), (conic xx, xy, yy, opacity), (rgb, 0).
[[vk::binding(7, 0)]] RWStructuredBuffer<float4> projected;
[[vk::binding(8, 0)]] RWStructuredBuffer<uint> keys;      // per instance
[[vk::binding(9, 0)]] RWStructuredBuffer<uint> values;    // per instance: the Gaussian
[[vk::binding(10, 0)]] RWStructuredBuffer<uint> counters; // [0]: instances emitted (may exceed the capacity)

[shader("compute")]
[numthreads(128, 1, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    const uint g = globalId.x;
    if (g >= pc.count) return;
    const float4x4 camera = cameras[viewIndex[0]];
    const float2 size = float2(pc.width, pc.height);
    const float3 position = means[g].xyz;
    const float3 mean = projectMean(position, camera, size);
    const float3 covariance = projectCovariance(position, logScales[g].xyz, rotations[g], camera, size);
    const float determinant = covariance.x * covariance.z - covariance.y * covariance.y;
    float radius = 0.0;
    int2 low = int2(0, 0), high = int2(-1, -1);
    if (mean.z > pc.nearDepth && determinant > 0.0) {
        radius = splatRadius(covariance);
        const int2 lastTile = int2(pc.tilesX, pc.tilesY) - 1;
        low = clamp(int2(floor((mean.xy - radius) / float(TILE))), int2(0, 0), lastTile);
        high = clamp(int2(floor((mean.xy + radius) / float(TILE))), int2(0, 0), lastTile);
        // Entirely off screen: no tile.
        if (any(mean.xy + radius < 0.0) || any(mean.xy - radius >= size)) high = int2(-1, -1);
    }
    const uint tileCount = all(high >= low) ? uint((high.x - low.x + 1) * (high.y - low.y + 1)) : 0u;
    if (tileCount == 0u) radius = 0.0;
    projected[3 * g] = float4(mean, radius);
    projected[3 * g + 1] = float4(radius > 0.0 ? conicOf(covariance) : float3(0.0), sigmoid(opacities[g]));
    projected[3 * g + 2] = float4(colors[g].rgb, 0.0);
    if (tileCount == 0u) return;
    uint base;
    InterlockedAdd(counters[0], tileCount, base);
    const uint depth = min(uint(mean.z * pc.depthScale), (1u << DEPTH_BITS) - 1u);
    uint k = base;
    for (int ty = low.y; ty <= high.y; ++ty) {
        for (int tx = low.x; tx <= high.x; ++tx, ++k) {
            if (k >= pc.capacity) return;
            keys[k] = ((uint(ty) * pc.tilesX + uint(tx)) << DEPTH_BITS) | depth;
            values[k] = g;
        }
    }
}
