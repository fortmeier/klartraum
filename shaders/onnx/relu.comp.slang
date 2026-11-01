// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

struct PushConstants {
    uint dimInput[4];  // [batch, channels, height, width]
    uint dimOutput[4]; // [batch, channels, height, width] - same as input for ReLU
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> input_data;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> output_data;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    uint index = globalId.x;
    uint total_elements = pc.dimInput[0] * pc.dimInput[1] * pc.dimInput[2] * pc.dimInput[3];
    if (index >= total_elements) {
        return;
    }
    // ReLU: f(x) = max(0, x)
    output_data[index] = max(0.0, input_data[index]);
}
