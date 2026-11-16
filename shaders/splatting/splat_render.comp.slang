// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Blends the depth-sorted splats of each tile front to back (one workgroup per tile, one invocation per pixel):
//   colour = sum_i c_i a_i T_i, T_(i+1) = T_i (1 - a_i), coverage = 1 - T_final,
// with a_i = min(splatAlpha, MAX_ALPHA), splats below MIN_ALPHA skipped and blending stopped before T would fall below
// MIN_TRANSMITTANCE. Writes premultiplied rgb and coverage, and per pixel how many of the tile's instances it went
// through (the backward kernel walks back from there).
import splat_common;

struct PushConstants {
    uint width;
    uint height;
    uint tilesX;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float4> projected; // three per Gaussian, see splat_project.comp.slang
[[vk::binding(1, 0)]] StructuredBuffer<uint> values;      // sorted instances: the Gaussian
[[vk::binding(2, 0)]] StructuredBuffer<uint2> ranges;     // per tile
[[vk::binding(3, 0)]] RWStructuredBuffer<float4> colors;  // per pixel
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> lastContributor; // per pixel

static const uint BATCH = TILE * TILE;
// The batch of splats the workgroup currently blends.
groupshared float4 batchMean[BATCH];
groupshared float4 batchConic[BATCH];
groupshared float4 batchColor[BATCH];

[shader("compute")]
[numthreads(TILE, TILE, 1)]
void main(uint3 groupId: SV_GroupID, uint3 localId: SV_GroupThreadID, uint localIndex: SV_GroupIndex) {
    const uint2 pixel = groupId.xy * TILE + localId.xy;
    const bool inside = pixel.x < pc.width && pixel.y < pc.height;
    const float2 centre = float2(pixel) + 0.5;
    const uint2 range = ranges[groupId.y * pc.tilesX + groupId.x];
    float3 color = 0.0;
    float transmittance = 1.0;
    bool blending = inside;
    uint last = 0u;
    for (uint start = range.x; start < range.y; start += BATCH) {
        if (start + localIndex < range.y) {
            const uint g = values[start + localIndex];
            batchMean[localIndex] = projected[3 * g];
            batchConic[localIndex] = projected[3 * g + 1];
            batchColor[localIndex] = projected[3 * g + 2];
        }
        GroupMemoryBarrierWithGroupSync();
        const uint n = min(BATCH, range.y - start);
        for (uint j = 0u; blending && j < n; ++j) {
            const float alpha = min(splatAlpha(batchMean[j].xy, batchConic[j].xyz, batchConic[j].w, centre), MAX_ALPHA);
            if (alpha < MIN_ALPHA) continue;
            const float next = transmittance * (1.0 - alpha);
            if (next < MIN_TRANSMITTANCE) {
                blending = false;
                break;
            }
            color += batchColor[j].rgb * alpha * transmittance;
            transmittance = next;
            last = start + j - range.x + 1u;
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (!inside) return;
    const uint index = pixel.y * pc.width + pixel.x;
    colors[index] = float4(color, 1.0 - transmittance);
    lastContributor[index] = last;
}
