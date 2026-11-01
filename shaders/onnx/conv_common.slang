// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Push constants shared by the NCHW 2D convolution shaders.
module conv_common;

public struct ConvPushConstants {
    // operation attributes
    public uint dilations[2];
    public uint groups[1];
    public uint kernel_shape[2];
    public uint pads[4];
    public uint strides[2];

    // tensor input sizes
    public uint dimInput[4];   // [N, C, H, W]
    public uint dimWeights[4]; // [output_channels, input_channels/groups, kH, kW]
    public uint dimBias[1];    // [output_channels]
    public uint dimOutput[4];  // inferred ONNX output shape
};
