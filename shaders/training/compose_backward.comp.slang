// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// The gradients of compose.comp.slang for the albedo's logit and the decoder's 4 raw outputs from the image's:
// d a_c = d image_c m s sigma'(a_c), d s = sum_c d image_c m sigmoid(a_c) 2 sigma'(s), d b_c = d image_c (1 - m)
// sigma'(b_c), with sigma'(x) = sigmoid(x) (1 - sigmoid(x)). One invocation per pixel.

struct PushConstants {
    uint batch;
    uint pixels;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> albedoLogit;      // [batch][3][pixels]
[[vk::binding(1, 0)]] StructuredBuffer<float> logits;           // [batch][4][pixels]
[[vk::binding(2, 0)]] StructuredBuffer<float> mask;             // [batch][pixels]
[[vk::binding(3, 0)]] StructuredBuffer<float> imageGradient;    // [batch][3][pixels]
[[vk::binding(4, 0)]] RWStructuredBuffer<float> albedoGradient; // [batch][3][pixels]
[[vk::binding(5, 0)]] RWStructuredBuffer<float> logitGradient;  // [batch][4][pixels]

float sigmoid(float x) { return 1.0 / (1.0 + exp(-x)); }

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 id: SV_DispatchThreadID) {
    if (id.x >= pc.batch * pc.pixels) return;
    const uint n = id.x / pc.pixels, p = id.x % pc.pixels;
    const float m = mask[id.x];
    const float sigmaS = sigmoid(logits[(n * 4u) * pc.pixels + p]);
    const float s = 2.0 * sigmaS;
    float shadingGradient = 0.0;
    for (uint c = 0u; c < 3u; ++c) {
        const float g = imageGradient[(n * 3u + c) * pc.pixels + p];
        const float a = sigmoid(albedoLogit[(n * 3u + c) * pc.pixels + p]);
        const float b = sigmoid(logits[(n * 4u + 1u + c) * pc.pixels + p]);
        albedoGradient[(n * 3u + c) * pc.pixels + p] = g * m * s * a * (1.0 - a);
        logitGradient[(n * 4u + 1u + c) * pc.pixels + p] = g * (1.0 - m) * b * (1.0 - b);
        shadingGradient += g * m * a;
    }
    logitGradient[(n * 4u) * pc.pixels + p] = shadingGradient * 2.0 * sigmaS * (1.0 - sigmaS);
}
