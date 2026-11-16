// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// RMS normalization of `width`-wide rows: y = x / sqrt(mean(x^2) + eps) * scale.
// Each token holds `rowsPerToken` rows; only the params[0] active tokens are
// normalized. One workgroup per row: its invocations sum strided squares,
// the subgroups and then shared memory add them up.
import subgroup;

struct PushConstants {
    uint width;
    uint rowsPerToken;
    float epsilon;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> X;
[[vk::binding(1, 0)]] StructuredBuffer<float> scale;
[[vk::binding(2, 0)]] StructuredBuffer<uint> params;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> Y;

// Each subgroup's sum of squares.
groupshared float partial[8];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 localId: SV_GroupThreadID, uint3 groupId: SV_GroupID) {
    const uint row = groupId.x;
    if (row >= params[0] * pc.rowsPerToken) return; // uniform per workgroup
    const uint offset = row * pc.width;
    float sum = 0.0;
    for (uint i = localId.x; i < pc.width; i += 256u) {
        const float value = X[offset + i];
        sum += value * value;
    }
    sum = WaveActiveSum(sum);
    if (WaveIsFirstLane()) partial[subgroupId()] = sum;
    GroupMemoryBarrierWithGroupSync();
    float total = 0.0;
    for (uint s = 0u; s < subgroupCount(); ++s) total += partial[s];
    const float inverse = rsqrt(total / float(pc.width) + pc.epsilon);
    for (uint i = localId.x; i < pc.width; i += 256u) Y[offset + i] = X[offset + i] * inverse * scale[i];
}
