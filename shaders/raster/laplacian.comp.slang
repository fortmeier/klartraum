// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// The uniform Laplacian of each vertex, L_v = p_v - mean of its neighbours' positions, and the smoothness loss
// weight x sum_v |L_v|^2 / vertex count (first pass of the Laplacian regularizer, laplacian_gradient.comp.slang the
// second).
import subgroup;

struct PushConstants {
    uint vertexCount;
    float weight;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float4> positions;      // xyz per vertex
[[vk::binding(1, 0)]] StructuredBuffer<uint> offsets;          // vertex count + 1 row starts
[[vk::binding(2, 0)]] StructuredBuffer<uint> neighbours;       // neighbour indices, row by row
[[vk::binding(3, 0)]] RWStructuredBuffer<float4> gradients;    // per vertex, added to
[[vk::binding(4, 0)]] RWStructuredBuffer<float> loss;          // one element, added to
[[vk::binding(5, 0)]] RWStructuredBuffer<float4> laplacians;   // per vertex, scratch

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    const uint v = globalId.x;
    float vertexLoss = 0.0;
    if (v < pc.vertexCount) {
        const uint begin = offsets[v], end = offsets[v + 1];
        float3 sum = 0.0;
        for (uint k = begin; k < end; ++k) sum += positions[neighbours[k]].xyz;
        const float3 laplacian = positions[v].xyz - sum / float(max(end - begin, 1u));
        laplacians[v] = float4(laplacian, 0.0);
        vertexLoss = pc.weight * dot(laplacian, laplacian) / float(pc.vertexCount);
    }
    const float groupLoss = WaveActiveSum(vertexLoss);
    if (WaveIsFirstLane()) InterlockedAdd(loss[0], groupLoss);
}
