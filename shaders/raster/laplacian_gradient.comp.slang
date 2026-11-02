// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// The gradient of the smoothness loss of laplacian.comp.slang: vertex v appears with weight 1 in its own Laplacian
// and with weight -1 / degree(n) in that of each neighbour n, so
//   dLoss / dp_v = 2 weight / vertex count x (L_v - sum over neighbours n of L_n / degree(n)).
// Each vertex adds its own gradient, so no atomics are needed.
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
    if (v >= pc.vertexCount) return;
    float3 gradient = laplacians[v].xyz;
    for (uint k = offsets[v]; k < offsets[v + 1]; ++k) {
        const uint n = neighbours[k];
        gradient -= laplacians[n].xyz / float(max(offsets[n + 1] - offsets[n], 1u));
    }
    gradients[v] += float4(2.0 * pc.weight / float(pc.vertexCount) * gradient, 0.0);
}
