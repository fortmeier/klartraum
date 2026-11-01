// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Pass 3 of the Winograd convolution: output transform and bias (see conv_winograd.slang).
import conv_winograd;

[[vk::binding(2, 0)]] StructuredBuffer<float> bias;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> outputData;
[[vk::binding(6, 0)]] StructuredBuffer<float> m;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint localIndex: SV_GroupIndex, uint3 groupId: SV_GroupID) {
    const uint2 grid = gridPosition(localIndex, groupId);
    for (uint item = grid.x; item < pc.outputChannels * tileCount(); item += grid.y) {
        transformOutput(item, m, bias, outputData);
    }
}
