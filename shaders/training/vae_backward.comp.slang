// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// The gradient of the statistics from the latents' gradient and the KL term of vae.comp.slang: d mu = d sample +
// w mu, d lv = d sample eps exp(lv / 2) / 2 + w (exp(lv) - 1) / 2, w = weight / batch for objects with pixels. One
// invocation per (sample, object).

struct PushConstants {
    uint batch;
    uint objects;
    uint size;     // latent numbers per object
    float weight;  // of the KL divergence, divided by the batch
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> statistics;      // [batch][objects][2 size]
[[vk::binding(1, 0)]] StructuredBuffer<float> epsilon;         // [batch][objects][size]
[[vk::binding(2, 0)]] StructuredBuffer<float> counts;          // [batch][objects]
[[vk::binding(3, 0)]] StructuredBuffer<float> latentGradient;  // [batch][objects][size]
[[vk::binding(4, 0)]] RWStructuredBuffer<float> statisticsGradient; // [batch][objects][2 size]

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID) {
    if (id.x >= pc.batch * pc.objects) return;
    const float w = counts[id.x] > 0.0 ? pc.weight / float(pc.batch) : 0.0;
    for (uint d = 0u; d < pc.size; ++d) {
        const float mu = statistics[id.x * 2u * pc.size + d], lv = statistics[id.x * 2u * pc.size + pc.size + d];
        const float g = latentGradient[id.x * pc.size + d];
        statisticsGradient[id.x * 2u * pc.size + d] = g + w * mu;
        statisticsGradient[id.x * 2u * pc.size + pc.size + d] =
            g * epsilon[id.x * pc.size + d] * 0.5 * exp(0.5 * lv) + w * 0.5 * (exp(lv) - 1.0);
    }
}
