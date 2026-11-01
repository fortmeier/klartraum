// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// WARNING: This file contains AI-generated code
// This compute shader was automatically generated and may require
// manual review and optimization for production use.
//
// Placeholder: the body does nothing. Reshape does not move data, the ONNX
// loader reinterprets the tensor instead.

struct PushConstants {
    uint dimInput[4]; // [batch, channels, height, width]
    uint dimShape[6]; // shape tensor dimensions (usually 1D with up to 6 elements)
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> input_data;
[[vk::binding(1, 0)]] StructuredBuffer<int> shape_data;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> output_data;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {}
