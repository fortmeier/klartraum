// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Column sums of a [rows, columns] matrix, e.g. a bias gradient over a
// batch. One invocation per column adds the rows in order, so the result does
// not depend on scheduling.

struct PushConstants {
    uint rows;
    uint columns;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> inputData;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> sums;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    const uint column = globalId.x;
    if (column >= pc.columns) return;
    float sum = 0.0;
    for (uint row = 0; row < pc.rows; ++row) sum += inputData[row * pc.columns + column];
    sums[column] = sum;
}
