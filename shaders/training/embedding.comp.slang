// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// The rows of a table (e.g. the appearance codes of the training images) picked by a batch's row indices.
// One invocation per element of the result.

struct PushConstants {
    uint batch;
    uint rows;
    uint size;
    float decay; // backward: the L2 penalty decay x mean over the batch of |code|^2
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> table; // [rows][size]
[[vk::binding(1, 0)]] StructuredBuffer<float> rows;  // [batch], row indices as floats
[[vk::binding(2, 0)]] RWStructuredBuffer<float> picked; // [batch][size]

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID) {
    if (id.x >= pc.batch * pc.size) return;
    const uint b = id.x / pc.size, d = id.x % pc.size;
    picked[id.x] = table[uint(rows[b] + 0.5) * pc.size + d];
}
