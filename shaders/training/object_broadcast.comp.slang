// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Every pixel gets its object's vector (0 off the objects), optionally divided by the object's pixel count: the latent
// G-buffer, and the backward pass of object_mean.comp.slang. Writes (or adds) the channels from targetOffset of the
// output. One invocation per output element of the written channels.

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

[[vk::binding(0, 0)]] StructuredBuffer<float> vectors;    // [batch][objects][channels]
[[vk::binding(1, 0)]] StructuredBuffer<float> objectMap;  // as for object_mean
[[vk::binding(2, 0)]] StructuredBuffer<float> counts;     // [batch][objects]
[[vk::binding(3, 0)]] RWStructuredBuffer<float> output;   // [batch][targetChannels][height][width]

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 id: SV_DispatchThreadID) {
    const uint pixels = pc.height * pc.width;
    if (id.x >= pc.batch * pc.channels * pixels) return;
    const uint p = id.x % pixels, c = (id.x / pixels) % pc.channels, n = id.x / (pixels * pc.channels);
    const uint k = objectOf(objectMap, n, p);
    float v = 0.0;
    if (k > 0u && k <= pc.objects) {
        v = vectors[(n * pc.objects + k - 1u) * pc.channels + c];
        if (pc.flag != 0u) v /= max(counts[n * pc.objects + k - 1u], 1.0);
    }
    const uint i = (n * pc.targetChannels + pc.targetOffset + c) * pixels + p;
    output[i] = pc.accumulate != 0u ? output[i] + v : v;
}
