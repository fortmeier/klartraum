// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// WARNING: This file contains AI-generated code
// This compute shader was automatically generated and may require
// manual review and optimization for production use.

import transpose_common;

[[vk::push_constant]] ConstantBuffer<TransposePushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> input_data;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> output_data;

// Helper function to convert multi-dimensional index to linear index
uint to_linear_index(uint b, uint d, uint h, uint w, uint batch_size, uint depth_size, uint height_size, uint width_size) {
    return b * (depth_size * height_size * width_size) +
           d * (height_size * width_size) +
           h * width_size +
           w;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    uint x = globalId.x;
    uint y = globalId.y;
    uint z = globalId.z;

    uint output_width = pc.dimOutput[3];
    uint output_height = pc.dimOutput[2];
    uint output_channels = pc.dimOutput[1];
    uint output_batch = pc.dimOutput[0];

    if (x >= output_width || y >= output_height || z >= output_channels) {
        return;
    }

    for (uint b = 0; b < output_batch; ++b) {
        int4 X = int4(b, z, y, x);
        int4 Q;
        Q[pc.dimPerm[0]] = X[0];
        Q[pc.dimPerm[1]] = X[1];
        Q[pc.dimPerm[2]] = X[2];
        Q[pc.dimPerm[3]] = X[3];

        uint input_idx = to_linear_index(Q[0], Q[1], Q[2], Q[3], pc.dimInput[0], pc.dimInput[1], pc.dimInput[2], pc.dimInput[3]);

        uint output_idx = to_linear_index(X[0], X[1], X[2], X[3], pc.dimOutput[0], pc.dimOutput[1], pc.dimOutput[2], pc.dimOutput[3]);

        // Calculate output tensor size based on transposed dimensions
        uint total_output_elements = pc.dimOutput[0] * pc.dimOutput[1] * pc.dimOutput[2] * pc.dimOutput[3];

        if (output_idx < total_output_elements) {
            output_data[output_idx] = input_data[input_idx];
        }
    }
}
