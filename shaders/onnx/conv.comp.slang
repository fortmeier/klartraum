// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// WARNING: This file contains AI-generated code
// This compute shader was automatically generated and may require
// manual review and optimization for production use.

import conv_common;

[[vk::push_constant]] ConstantBuffer<ConvPushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> input_data;
[[vk::binding(1, 0)]] StructuredBuffer<float> weight_data;
[[vk::binding(2, 0)]] StructuredBuffer<float> bias_data;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> output_data;

// Helper function to calculate tensor index for NCHW layout
uint getTensorIndex4D(uint n, uint c, uint h, uint w, uint N, uint C, uint H, uint W) {
    return n * (C * H * W) + c * (H * W) + h * W + w;
}

// Helper function to calculate weight index for [out_channels, in_channels/groups, kH, kW]
uint getWeightIndex4D(uint out_c, uint in_c, uint kh, uint kw, uint OUT_C, uint IN_C, uint KH, uint KW) {
    return out_c * (IN_C * KH * KW) + in_c * (KH * KW) + kh * KW + kw;
}

// Basic 2D convolution implementation
[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    uint gx = globalId.x;
    uint gy = globalId.y;
    uint gz = globalId.z;

    // Extract tensor dimensions from push constants
    uint input_N = pc.dimInput[0]; // batch size
    uint input_C = pc.dimInput[1]; // input channels
    uint input_H = pc.dimInput[2]; // input height
    uint input_W = pc.dimInput[3]; // input width

    uint weight_OUT_C = pc.dimWeights[0]; // output channels
    uint weight_IN_C = pc.dimWeights[1];  // input channels / groups
    uint weight_KH = pc.dimWeights[2];    // kernel height
    uint weight_KW = pc.dimWeights[3];    // kernel width

    // Extract convolution parameters
    uint kernel_h = pc.kernel_shape[0];
    uint kernel_w = pc.kernel_shape[1];
    uint pad_top = pc.pads[0];
    uint pad_left = pc.pads[1];
    uint stride_h = pc.strides[0];
    uint stride_w = pc.strides[1];
    uint dilation_h = pc.dilations[0];
    uint dilation_w = pc.dilations[1];
    uint groups = pc.groups[0];

    // Calculate output dimensions
    uint output_H = pc.dimOutput[2] == 0
        ? (input_H + 2 * pad_top - dilation_h * (kernel_h - 1) - 1) / stride_h + 1
        : pc.dimOutput[2];
    uint output_W = pc.dimOutput[3] == 0
        ? (input_W + 2 * pad_left - dilation_w * (kernel_w - 1) - 1) / stride_w + 1
        : pc.dimOutput[3];

    // Z encodes both batch and output channel.
    if (gx >= output_W || gy >= output_H || gz >= input_N * weight_OUT_C) {
        return; // Out of bounds
    }

    uint batch = gz / weight_OUT_C;
    uint out_channel = gz % weight_OUT_C;
    uint out_y = gy;
    uint out_x = gx;

    float sum = 0.0;

    // Calculate input channels per group
    uint in_channels_per_group = input_C / groups;
    uint group_id = out_channel / (weight_OUT_C / groups);
    uint group_start_channel = group_id * in_channels_per_group;

    // Convolution computation
    for (uint in_c = 0; in_c < in_channels_per_group; ++in_c) {
        uint actual_in_c = group_start_channel + in_c;

        for (uint kh = 0; kh < kernel_h; ++kh) {
            for (uint kw = 0; kw < kernel_w; ++kw) {
                // Calculate input position with padding and dilation
                int input_y = int(out_y * stride_h + kh * dilation_h) - int(pad_top);
                int input_x = int(out_x * stride_w + kw * dilation_w) - int(pad_left);

                // Check bounds
                if (input_y >= 0 && input_y < int(input_H) &&
                    input_x >= 0 && input_x < int(input_W)) {

                    // Get input value
                    uint input_idx = getTensorIndex4D(batch, actual_in_c, uint(input_y), uint(input_x),
                                                      input_N, input_C, input_H, input_W);

                    // Get weight value
                    uint weight_idx = getWeightIndex4D(out_channel, in_c, kh, kw,
                                                       weight_OUT_C, weight_IN_C, weight_KH, weight_KW);

                    // Bounds checking for buffer access
                    if (input_idx < input_data.getCount() && weight_idx < weight_data.getCount()) {
                        sum += input_data[input_idx] * weight_data[weight_idx];
                    }
                }
            }
        }
    }

    // Add bias if available
    if (pc.dimBias[0] > 0 && out_channel < bias_data.getCount()) {
        sum += bias_data[out_channel];
    }

    // Write output
    uint output_idx = getTensorIndex4D(batch, out_channel, out_y, out_x,
                                       input_N, weight_OUT_C, output_H, output_W);

    if (output_idx < output_data.getCount()) {
        output_data[output_idx] = sum;
    }
}
