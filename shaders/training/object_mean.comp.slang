// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// The mean (or sum) of a feature map's values over each object's pixels: one workgroup per (sample, object, channel)
// adds the values of the object's pixels in a fixed order; the first channel's workgroup writes the pixel count.
// Objects without pixels get 0.

struct PushConstants {
    uint batch;
    uint channels;     // of the values (object_mean) or of each object's vector (object_broadcast)
    uint height;       // of the values' pixel grid
    uint width;
    uint objects;      // objects 1..objects (0: none)
    uint objectsWidth; // of the object map, which may be finer: the values' pixel (y, x) takes its object at
    uint objectsHeight; //   (y step + offset, x step + offset)
    uint step;
    uint offset;
    uint flag;          // object_mean: 1 mean, 0 sum; object_broadcast: 1 divide by the object's pixel count
    uint targetChannels; // object_broadcast: channels of the output tensor
    uint targetOffset;   //   and the first one written
    uint accumulate;     //   1: add to it
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

uint objectOf(StructuredBuffer<float> objectMap, uint n, uint p) {
    const uint y = p / pc.width, x = p % pc.width;
    const uint i = (y * pc.step + pc.offset) * pc.objectsWidth + x * pc.step + pc.offset;
    return uint(objectMap[n * pc.objectsWidth * pc.objectsHeight + i] + 0.5);
}

[[vk::binding(0, 0)]] StructuredBuffer<float> values;     // [batch][channels][height][width]
[[vk::binding(1, 0)]] StructuredBuffer<float> objectMap;  // [batch][objectsHeight][objectsWidth], indices as floats
[[vk::binding(2, 0)]] RWStructuredBuffer<float> means;    // [batch][objects][channels]
[[vk::binding(3, 0)]] RWStructuredBuffer<float> counts;   // [batch][objects]

groupshared float partialSum[256];
groupshared float partialCount[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group: SV_GroupID, uint3 local: SV_GroupThreadID) {
    const uint c = group.x % pc.channels, k = (group.x / pc.channels) % pc.objects;
    const uint n = group.x / (pc.channels * pc.objects);
    const uint pixels = pc.height * pc.width;
    float sum = 0.0, count = 0.0;
    for (uint p = local.x; p < pixels; p += 256u)
        if (objectOf(objectMap, n, p) == k + 1u) {
            sum += values[(n * pc.channels + c) * pixels + p];
            count += 1.0;
        }
    partialSum[local.x] = sum;
    partialCount[local.x] = count;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (local.x < stride) {
            partialSum[local.x] += partialSum[local.x + stride];
            partialCount[local.x] += partialCount[local.x + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (local.x == 0u) {
        const float total = partialCount[0];
        means[(n * pc.objects + k) * pc.channels + c] = pc.flag != 0u ? partialSum[0] / max(total, 1.0) : partialSum[0];
        if (c == 0u) counts[n * pc.objects + k] = total;
    }
}
