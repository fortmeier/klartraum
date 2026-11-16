// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Copies `count` channels of an NCHW tensor into another one at a channel offset, or adds them there: a concat
// along the channels (copy each part to its offset) and its backward pass (copy each part's channels back out, adding
// where a tensor's gradient comes from several places, e.g. a U-Net skip). One invocation per copied element.

struct PushConstants {
    uint batch;
    uint pixels;
    uint sourceChannels;
    uint sourceOffset;
    uint targetChannels;
    uint targetOffset;
    uint count;
    uint accumulate; // 1: add to the target
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> source;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> target;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 id: SV_DispatchThreadID) {
    if (id.x >= pc.batch * pc.count * pc.pixels) return;
    const uint p = id.x % pc.pixels;
    const uint c = (id.x / pc.pixels) % pc.count;
    const uint n = id.x / (pc.pixels * pc.count);
    const float value = source[(n * pc.sourceChannels + pc.sourceOffset + c) * pc.pixels + p];
    const uint t = (n * pc.targetChannels + pc.targetOffset + c) * pc.pixels + p;
    target[t] = pc.accumulate != 0u ? target[t] + value : value;
}
