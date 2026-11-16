// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// ChannelNorm (train_photo_enhancer.py): every pixel of an NCHW tensor normalized over its channels, then scaled
// and shifted per channel. Keeps each pixel's mean and 1 / standard deviation for the backward pass. One invocation
// per pixel.

struct PushConstants {
    uint batch;
    uint channels;
    uint pixels;
    float epsilon;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> x;      // [batch][channels][pixels]
[[vk::binding(1, 0)]] StructuredBuffer<float> weight; // [channels]
[[vk::binding(2, 0)]] StructuredBuffer<float> bias;   // [channels]
[[vk::binding(3, 0)]] RWStructuredBuffer<float> y;    // [batch][channels][pixels]
[[vk::binding(4, 0)]] RWStructuredBuffer<float> stats; // [batch][2][pixels]: mean, 1 / standard deviation

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 id: SV_DispatchThreadID) {
    if (id.x >= pc.batch * pc.pixels) return;
    const uint n = id.x / pc.pixels, p = id.x % pc.pixels;
    const uint base = n * pc.channels * pc.pixels + p;
    float mean = 0.0;
    for (uint c = 0u; c < pc.channels; ++c) mean += x[base + c * pc.pixels];
    mean /= float(pc.channels);
    float variance = 0.0;
    for (uint c = 0u; c < pc.channels; ++c) {
        const float d = x[base + c * pc.pixels] - mean;
        variance += d * d;
    }
    const float inverse = 1.0 / sqrt(variance / float(pc.channels) + pc.epsilon);
    for (uint c = 0u; c < pc.channels; ++c)
        y[base + c * pc.pixels] = (x[base + c * pc.pixels] - mean) * inverse * weight[c] + bias[c];
    stats[(n * 2u) * pc.pixels + p] = mean;
    stats[(n * 2u + 1u) * pc.pixels + p] = inverse;
}
