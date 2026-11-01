// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Pass 1 of the Winograd convolution: weight and input transforms (see conv_winograd.slang).
import conv_winograd;

[[vk::binding(0, 0)]] StructuredBuffer<float> inputData;
[[vk::binding(1, 0)]] StructuredBuffer<float> weights;
[[vk::binding(4, 0)]] RWStructuredBuffer<float> u;
[[vk::binding(5, 0)]] RWStructuredBuffer<float> v;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint localIndex: SV_GroupIndex, uint3 groupId: SV_GroupID) {
    const uint2 grid = gridPosition(localIndex, groupId);
    // The weights are transformed once, in the first chunk; later chunks reuse them.
    const uint weightItems = pc.frameBegin == 0 ? pc.outputChannels * reductionCount() : 0;
    const uint inputItems = reductionCount() * tileCount();
    for (uint item = grid.x; item < weightItems + inputItems; item += grid.y) {
        if (item < weightItems) {
            transformWeights(item, weights, u);
        } else {
            transformInput(item - weightItems, inputData, v);
        }
    }
}
