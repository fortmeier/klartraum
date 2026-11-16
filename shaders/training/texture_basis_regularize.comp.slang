// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// The regularisation of one level of the code-mixed feature texture (texture_basis.comp.slang): the gradient of
// decay x sum of squared texels + smoothness x sum of squared differences between neighbouring texels of the same
// plane (right and down, none across the plane's edges), added to the table's gradient. One invocation per texel.

struct RegularizeConstants {
    uint planes;     // faces x layers x channels
    uint resolution; // texels per side
    float decay;
    float smoothness;
};
[[vk::push_constant]] ConstantBuffer<RegularizeConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> table;           // [planes][resolution][resolution]
[[vk::binding(1, 0)]] RWStructuredBuffer<float> tableGradient; // the table's layout

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 id: SV_DispatchThreadID) {
    const uint texels = pc.resolution * pc.resolution;
    if (id.x >= pc.planes * texels) return;
    const uint t = id.x % texels, x = t % pc.resolution, y = t / pc.resolution;
    const float value = table[id.x];
    float difference = 0.0;
    if (x > 0u) difference += value - table[id.x - 1u];
    if (x + 1u < pc.resolution) difference += value - table[id.x + 1u];
    if (y > 0u) difference += value - table[id.x - pc.resolution];
    if (y + 1u < pc.resolution) difference += value - table[id.x + pc.resolution];
    tableGradient[id.x] += 2.0 * pc.decay * value + 2.0 * pc.smoothness * difference;
}
