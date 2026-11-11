// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// The VAE's latents from the encoder's per-object statistics (means, then log-variances): sample = mu + exp(lv / 2)
// eps, and each object's share of weight x KL(N(mu, sigma) || N(0, 1)) / batch (0 for objects without pixels). One
// invocation per (sample, object).

struct PushConstants {
    uint batch;
    uint objects;
    uint size;     // latent numbers per object
    float weight;  // of the KL divergence, divided by the batch
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> statistics; // [batch][objects][2 size]
[[vk::binding(1, 0)]] StructuredBuffer<float> epsilon;    // [batch][objects][size]
[[vk::binding(2, 0)]] StructuredBuffer<float> counts;     // [batch][objects]
[[vk::binding(3, 0)]] RWStructuredBuffer<float> latents;  // [batch][objects][size]
[[vk::binding(4, 0)]] RWStructuredBuffer<float> shares;   // [batch][objects]

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID) {
    if (id.x >= pc.batch * pc.objects) return;
    float kl = 0.0;
    for (uint d = 0u; d < pc.size; ++d) {
        const float mu = statistics[id.x * 2u * pc.size + d], lv = statistics[id.x * 2u * pc.size + pc.size + d];
        latents[id.x * pc.size + d] = mu + exp(0.5 * lv) * epsilon[id.x * pc.size + d];
        kl += -0.5 * (1.0 + lv - mu * mu - exp(lv));
    }
    shares[id.x] = counts[id.x] > 0.0 ? pc.weight * kl / float(pc.batch) : 0.0;
}
