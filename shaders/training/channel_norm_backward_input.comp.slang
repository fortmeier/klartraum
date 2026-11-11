// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// The input gradient of ChannelNorm: with g = dy * weight and the normalized x^, dx = (g - mean(g) - x^ mean(g x^))
// / standard deviation, the means over the pixel's channels. One invocation per pixel.

struct PushConstants {
    uint batch;
    uint channels;
    uint pixels;
    float epsilon;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> x;
[[vk::binding(1, 0)]] StructuredBuffer<float> weight;
[[vk::binding(2, 0)]] StructuredBuffer<float> stats;
[[vk::binding(3, 0)]] StructuredBuffer<float> outputGradient;
[[vk::binding(4, 0)]] RWStructuredBuffer<float> inputGradient;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 id: SV_DispatchThreadID) {
    if (id.x >= pc.batch * pc.pixels) return;
    const uint n = id.x / pc.pixels, p = id.x % pc.pixels;
    const uint base = n * pc.channels * pc.pixels + p;
    const float mean = stats[(n * 2u) * pc.pixels + p], inverse = stats[(n * 2u + 1u) * pc.pixels + p];
    float meanG = 0.0, meanGX = 0.0;
    for (uint c = 0u; c < pc.channels; ++c) {
        const uint i = base + c * pc.pixels;
        const float g = outputGradient[i] * weight[c];
        meanG += g;
        meanGX += g * (x[i] - mean) * inverse;
    }
    meanG /= float(pc.channels);
    meanGX /= float(pc.channels);
    for (uint c = 0u; c < pc.channels; ++c) {
        const uint i = base + c * pc.pixels;
        const float g = outputGradient[i] * weight[c];
        inputGradient[i] = (g - meanG - (x[i] - mean) * inverse * meanGX) * inverse;
    }
}
