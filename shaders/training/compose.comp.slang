// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// The albedo renderer's image (latent_renderer.compose()): from the albedo's logit a, the decoder's 4 raw outputs
// (shading s, background b) and the object mask m, image = m sigmoid(a) 2 sigmoid(s) + (1 - m) sigmoid(b); also the
// albedo sigmoid(a) and the shading 2 sigmoid(s) on their own. One invocation per pixel.

struct PushConstants {
    uint batch;
    uint pixels;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> albedoLogit; // [batch][3][pixels]
[[vk::binding(1, 0)]] StructuredBuffer<float> logits;      // [batch][4][pixels]: shading, background RGB
[[vk::binding(2, 0)]] StructuredBuffer<float> mask;        // [batch][pixels]
[[vk::binding(3, 0)]] RWStructuredBuffer<float> image;     // [batch][3][pixels]
[[vk::binding(4, 0)]] RWStructuredBuffer<float> albedo;    // [batch][3][pixels]
[[vk::binding(5, 0)]] RWStructuredBuffer<float> shading;   // [batch][pixels]

float sigmoid(float x) { return 1.0 / (1.0 + exp(-x)); }

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 id: SV_DispatchThreadID) {
    if (id.x >= pc.batch * pc.pixels) return;
    const uint n = id.x / pc.pixels, p = id.x % pc.pixels;
    const float m = mask[id.x];
    const float s = 2.0 * sigmoid(logits[(n * 4u) * pc.pixels + p]);
    shading[id.x] = s;
    for (uint c = 0u; c < 3u; ++c) {
        const float a = sigmoid(albedoLogit[(n * 3u + c) * pc.pixels + p]);
        const float b = sigmoid(logits[(n * 4u + 1u + c) * pc.pixels + p]);
        albedo[(n * 3u + c) * pc.pixels + p] = a;
        image[(n * 3u + c) * pc.pixels + p] = m * a * s + (1.0 - m) * b;
    }
}
