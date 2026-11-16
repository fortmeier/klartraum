// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// The table gradient of embedding.comp.slang, with the codes' L2 penalty decay x mean_b |code_b|^2: every row adds
// the gradients of the batch entries that picked it, and 2 decay code / batch for each of them. Rows nobody picked get
// zero. One invocation per table element, looping over the batch (no atomics).

struct PushConstants {
    uint batch;
    uint rows;
    uint size;
    float decay; // backward: the L2 penalty decay x mean over the batch of |code|^2
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> table;          // [rows][size]
[[vk::binding(1, 0)]] StructuredBuffer<float> rows;           // [batch]
[[vk::binding(2, 0)]] StructuredBuffer<float> pickedGradient; // [batch][size]
[[vk::binding(3, 0)]] RWStructuredBuffer<float> tableGradient; // [rows][size]

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID) {
    if (id.x >= pc.rows * pc.size) return;
    const uint row = id.x / pc.size, d = id.x % pc.size;
    float sum = 0.0;
    for (uint b = 0u; b < pc.batch; ++b)
        if (uint(rows[b] + 0.5) == row)
            sum += pickedGradient[b * pc.size + d] + 2.0 * pc.decay * table[id.x] / float(pc.batch);
    tableGradient[id.x] = sum;
}
