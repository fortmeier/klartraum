// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// A group-one NCHW 1x1 convolution is a matrix multiplication between
// [output channels, input channels] weights and [input channels, spatial]
// activations. Tile both operands to reuse them across 16 outputs.

import conv_common;

[[vk::push_constant]] ConstantBuffer<ConvPushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> inputData;
[[vk::binding(1, 0)]] StructuredBuffer<float> weights;
[[vk::binding(2, 0)]] StructuredBuffer<float> bias;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> outputData;

// The weights of the tile's 16 output channels for 16 input channels.
groupshared float weightTile[16][16];
// The activations of 16 input channels at the tile's 16 positions.
groupshared float inputTile[16][16];

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 globalId: SV_DispatchThreadID, uint3 localId: SV_GroupThreadID) {
    const uint spatial = pc.dimOutput[2] * pc.dimOutput[3];
    const uint spatialIndex = globalId.x;
    const uint outputChannel = globalId.y;
    const uint batch = globalId.z;
    const uint inputChannels = pc.dimInput[1];
    const uint outputChannels = pc.dimOutput[1];
    float value = 0.0;

    for (uint tile = 0; tile < inputChannels; tile += 16) {
        const uint weightInputChannel = tile + localId.x;
        const uint inputChannel = tile + localId.y;
        weightTile[localId.y][localId.x] =
            outputChannel < outputChannels && weightInputChannel < inputChannels
                ? weights[outputChannel * inputChannels + weightInputChannel]
                : 0.0;
        inputTile[localId.y][localId.x] =
            inputChannel < inputChannels && spatialIndex < spatial
                ? inputData[(batch * inputChannels + inputChannel) * spatial + spatialIndex]
                : 0.0;
        GroupMemoryBarrierWithGroupSync();
        for (uint channel = 0; channel < 16; ++channel) {
            value += weightTile[localId.y][channel] * inputTile[channel][localId.x];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (outputChannel < outputChannels && spatialIndex < spatial) {
        if (pc.dimBias[0] > 0) value += bias[outputChannel];
        outputData[(batch * outputChannels + outputChannel) * spatial + spatialIndex] = value;
    }
}
