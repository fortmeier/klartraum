// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Start of LCM inpainting (sampleInpainted): the clean latent to keep is the
// encoder's latent times the VAE scaling factor; the start latent is the
// noise where the mask generates and the noised known latent where it keeps.
// Also sets the timestep of the first denoising step.

struct PushConstants {
    uint count;  // values of a latent, channels * plane
    uint plane;  // latent pixels
    float scale; // VAE scaling factor
    float sqrtAlpha;
    float sqrtOneMinusAlpha;
    uint timestepLow;
    uint timestepHigh;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> encoded;
[[vk::binding(1, 0)]] StructuredBuffer<float> mask;
[[vk::binding(2, 0)]] StructuredBuffer<float> noise; // [steps][count]
[[vk::binding(3, 0)]] RWStructuredBuffer<float> known;
[[vk::binding(4, 0)]] RWStructuredBuffer<float> latents;
[[vk::binding(5, 0)]] RWStructuredBuffer<uint> timestep; // one int64

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    uint i = globalId.x;
    if (i == 0) {
        timestep[0] = pc.timestepLow;
        timestep[1] = pc.timestepHigh;
    }
    if (i >= pc.count) {
        return;
    }
    float clean = encoded[i] * pc.scale;
    known[i] = clean;
    float m = mask[i % pc.plane];
    float n = noise[i];
    float outside = pc.sqrtAlpha * clean + pc.sqrtOneMinusAlpha * n;
    latents[i] = m * n + (1.0 - m) * outside;
}
