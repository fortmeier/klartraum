// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Specialized NCHW 3x3 convolution for the stride-one, pad-one, group-one
// case used by Stable Diffusion residual blocks. Four output channels share
// the 10x10 input footprint needed by an 8x8 output tile.

import conv_common;

[[vk::push_constant]] ConstantBuffer<ConvPushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> inputData;
[[vk::binding(1, 0)]] StructuredBuffer<float> weights;
[[vk::binding(2, 0)]] StructuredBuffer<float> bias;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> outputData;

// The 10x10 input footprint of the 8x8 output tile for one input channel.
groupshared float inputTile[100];

[shader("compute")]
[numthreads(8, 8, 4)]
void main(uint3 globalId: SV_DispatchThreadID, uint3 localId: SV_GroupThreadID, uint3 groupId: SV_GroupID) {
    const uint outputWidth = pc.dimOutput[3];
    const uint outputHeight = pc.dimOutput[2];
    const uint outputChannels = pc.dimOutput[1];
    const uint inputChannels = pc.dimInput[1];
    const uint inputHeight = pc.dimInput[2];
    const uint inputWidth = pc.dimInput[3];
    const uint channelGroupCount = (outputChannels + 3) / 4;
    const uint batch = groupId.z / channelGroupCount;
    const uint outputChannel = (groupId.z % channelGroupCount) * 4 + localId.z;
    const uint outputX = globalId.x;
    const uint outputY = globalId.y;
    const bool outputIsValid = outputChannel < outputChannels && outputX < outputWidth && outputY < outputHeight;
    const uint localIndex = localId.z * 64 + localId.y * 8 + localId.x;
    const uint inputPlane = inputHeight * inputWidth;
    const uint inputBatchOffset = batch * inputChannels * inputPlane;
    float value = 0.0;

    for (uint inputChannel = 0; inputChannel < inputChannels; ++inputChannel) {
        for (uint tileIndex = localIndex; tileIndex < 100; tileIndex += 256) {
            const int inputY = int(groupId.y * 8 + tileIndex / 10) - 1;
            const int inputX = int(groupId.x * 8 + tileIndex % 10) - 1;
            if (inputY >= 0 && inputY < int(inputHeight) && inputX >= 0 && inputX < int(inputWidth)) {
                inputTile[tileIndex] =
                    inputData[inputBatchOffset + inputChannel * inputPlane + uint(inputY) * inputWidth + uint(inputX)];
            } else {
                inputTile[tileIndex] = 0.0;
            }
        }
        GroupMemoryBarrierWithGroupSync();

        if (outputIsValid) {
            const uint weightOffset = (outputChannel * inputChannels + inputChannel) * 9;
            const uint tileOffset = localId.y * 10 + localId.x;
            for (uint kernelY = 0; kernelY < 3; ++kernelY) {
                for (uint kernelX = 0; kernelX < 3; ++kernelX) {
                    value += inputTile[tileOffset + kernelY * 10 + kernelX] *
                             weights[weightOffset + kernelY * 3 + kernelX];
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (outputIsValid) {
        if (pc.dimBias[0] > 0) value += bias[outputChannel];
        outputData[((batch * outputChannels + outputChannel) * outputHeight + outputY) * outputWidth + outputX] = value;
    }
}
