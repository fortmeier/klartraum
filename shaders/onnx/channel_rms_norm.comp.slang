// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// RMS normalization over the channel axis of an N x C x P tensor (the Wan VAE's
// channel RMSNorm), optionally followed by SiLU. One invocation per (n, p): it
// reads its C values twice (sum of squares, then the normalized output), and
// neighbouring invocations read neighbouring positions.

struct PushConstants {
    uint batch;
    uint channels;
    uint positions;
    uint silu;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> inputData;
[[vk::binding(1, 0)]] StructuredBuffer<float> eps;
[[vk::binding(2, 0)]] StructuredBuffer<float> scale;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> outputData;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    const uint position = globalId.x;
    const uint n = globalId.y;
    if (position >= pc.positions || n >= pc.batch) return;
    const uint base = n * pc.channels * pc.positions + position;
    float sum = 0.0;
    for (uint c = 0; c < pc.channels; ++c) {
        const float value = inputData[base + c * pc.positions];
        sum += value * value;
    }
    const float inverse = rsqrt(sum / float(pc.channels) + eps[0]);
    for (uint c = 0; c < pc.channels; ++c) {
        float y = inputData[base + c * pc.positions] * inverse * scale[c];
        if (pc.silu != 0) y = y / (1.0 + exp(-y));
        outputData[base + c * pc.positions] = y;
    }
}
