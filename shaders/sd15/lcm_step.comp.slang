// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// One step of LCM inpainting (LcmScheduler::step followed by the latent
// blending of sampleInpainted), in place on the latents the denoiser reads.
// Before the last step it noises the result to the next timestep and sets that
// timestep; after the last step it writes the decoder's input instead.

struct PushConstants {
    uint count; // values of a latent, channels * plane
    uint plane; // latent pixels
    uint next;  // index of the next step's noise
    uint last;  // 1 for the last step
    float sqrtAlpha;
    float sqrtOneMinusAlpha;
    float sqrtAlphaNext;
    float sqrtOneMinusAlphaNext;
    float cSkip;
    float cOut;
    float sqrtAlphaKnown; // alpha of the next step's timestep, for noising the known latent
    float sqrtOneMinusAlphaKnown;
    float scale; // VAE scaling factor
    uint timestepLow;
    uint timestepHigh;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] RWStructuredBuffer<float> latents;
[[vk::binding(1, 0)]] StructuredBuffer<float> prediction;
[[vk::binding(2, 0)]] StructuredBuffer<float> known;
[[vk::binding(3, 0)]] StructuredBuffer<float> mask;
[[vk::binding(4, 0)]] StructuredBuffer<float> noise;     // [steps][count]
[[vk::binding(5, 0)]] RWStructuredBuffer<uint> timestep; // one int64
[[vk::binding(6, 0)]] RWStructuredBuffer<float> decoderInput;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    uint i = globalId.x;
    if (i == 0 && pc.last == 0) {
        timestep[0] = pc.timestepLow;
        timestep[1] = pc.timestepHigh;
    }
    if (i >= pc.count) {
        return;
    }
    float x = latents[i];
    float original = (x - pc.sqrtOneMinusAlpha * prediction[i]) / pc.sqrtAlpha;
    float denoised = pc.cOut * original + pc.cSkip * x;
    float m = mask[i % pc.plane];
    if (pc.last != 0) {
        float value = m * denoised + (1.0 - m) * known[i];
        latents[i] = value;
        decoderInput[i] = value / pc.scale;
        return;
    }
    float n = noise[pc.next * pc.count + i];
    float noised = pc.sqrtAlphaNext * denoised + pc.sqrtOneMinusAlphaNext * n;
    float outside = pc.sqrtAlphaKnown * known[i] + pc.sqrtOneMinusAlphaKnown * n;
    latents[i] = m * noised + (1.0 - m) * outside;
}
