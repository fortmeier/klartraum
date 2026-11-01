// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// WARNING: This file contains AI-generated code
// This compute shader was automatically generated and may require
// manual review and optimization for production use.

struct PushConstants {
    uint dimInput[4];   // [batch, input_channels, input_height, input_width]
    uint dimWeights[4]; // [input_channels, output_channels, kernel_height, kernel_width]
    uint dimOutput[4];  // [batch, output_channels, output_height, output_width]
    uint dimBias[4];    // [output_channels] (only first element used)
    uint kernel_shape[8];
    uint strides[8];
    uint pads[8];
    uint dilations[8];
    uint groups[8];
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> input_data;
[[vk::binding(1, 0)]] StructuredBuffer<float> weights_data;
[[vk::binding(2, 0)]] StructuredBuffer<float> bias_data;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> output_data;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    uint x = globalId.x;
    uint y = globalId.y;
    uint z = globalId.z;

    uint output_batch = pc.dimOutput[0];
    uint output_channels = pc.dimOutput[1];
    uint output_height = pc.dimOutput[2];
    uint output_width = pc.dimOutput[3];

    uint input_channels = pc.dimInput[1];
    uint input_height = pc.dimInput[2];
    uint input_width = pc.dimInput[3];

    uint kernel_height = pc.kernel_shape[0];
    uint kernel_width = pc.kernel_shape[1];
    uint stride_h = pc.strides[0];
    uint stride_w = pc.strides[1];
    uint pad_h = pc.pads[0];
    uint pad_w = pc.pads[1];

    if (x >= output_width || y >= output_height || z >= output_channels) {
        return;
    }

    for (uint b = 0; b < output_batch; ++b) {
        float sum = 0.0;

        // Transposed convolution algorithm:
        // For each output position (x, y), we need to find all input positions
        // whose kernel overlaps with this output position.
        //
        // The relationship is: output[out_y][out_x] gets contributions from
        // input[in_y][in_x] * kernel[ky][kx] where:
        // out_y = in_y * stride_h + ky - pad_h
        // out_x = in_x * stride_w + kx - pad_w
        //
        // Solving for input position:
        // in_y = (out_y + pad_h - ky) / stride_h
        // in_x = (out_x + pad_w - kx) / stride_w

        for (uint ic = 0; ic < input_channels; ++ic) {
            for (uint ky = 0; ky < kernel_height; ++ky) {
                for (uint kx = 0; kx < kernel_width; ++kx) {
                    // Calculate which input position would contribute through this kernel position
                    int in_y_num = int(y + pad_h) - int(ky);
                    int in_x_num = int(x + pad_w) - int(kx);

                    // Check if input position is valid (must be on stride grid and in bounds)
                    if (in_y_num >= 0 && in_x_num >= 0 &&
                        (in_y_num % int(stride_h)) == 0 && (in_x_num % int(stride_w)) == 0) {

                        uint in_y = uint(in_y_num / int(stride_h));
                        uint in_x = uint(in_x_num / int(stride_w));

                        if (in_y < input_height && in_x < input_width) {
                            uint input_idx = b * input_channels * input_height * input_width +
                                             ic * input_height * input_width +
                                             in_y * input_width + in_x;

                            // For transposed conv, weight layout is [input_channels, output_channels, kh, kw]
                            uint weight_idx = ic * output_channels * kernel_height * kernel_width +
                                              z * kernel_height * kernel_width +
                                              ky * kernel_width + kx;

                            sum += input_data[input_idx] * weights_data[weight_idx];
                        }
                    }
                }
            }
        }

        // Add bias if available
        if (pc.dimBias[0] > 0) {
            sum += bias_data[z];
        }

        uint output_idx = b * output_channels * output_height * output_width +
                          z * output_height * output_width +
                          y * output_width + x;

        output_data[output_idx] = sum;
    }
}
