// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import transpose_common;

[[vk::push_constant]] ConstantBuffer<TransposePushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> input_data;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> output_data;

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    uint x = globalId.x;
    uint y = globalId.y;

    uint output_dim0 = pc.dimOutput[0];
    uint output_dim1 = pc.dimOutput[1];

    if (x >= output_dim1 || y >= output_dim0) {
        return;
    }

    // Build output index
    uint out_idx = y * output_dim1 + x;

    // Permute indices
    uint in_idx = 0;
    if (pc.dimPerm[0] == 1 && pc.dimPerm[1] == 0) {
        in_idx = x * output_dim0 + y;
    } else {
        in_idx = out_idx; // No permutation
    }

    output_data[out_idx] = input_data[in_idx];
}
