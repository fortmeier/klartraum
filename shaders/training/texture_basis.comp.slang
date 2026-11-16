// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// The code-mixed feature texture (latent_renderer.TextureBasis): every textured pixel's features are its face's base
// layer plus each further layer times the pixel's latent, sampled bilinearly at its (u, v); zero (or nothing added)
// elsewhere. One invocation per pixel.
import texture_basis;

[[vk::push_constant]] ConstantBuffer<TextureBasisConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> coordinates; // [batch][3][pixels]
[[vk::binding(1, 0)]] StructuredBuffer<float> classes;     // [batch][pixels]
[[vk::binding(2, 0)]] StructuredBuffer<float> table;       // [faces][layers][channels][resolution][resolution]
[[vk::binding(3, 0)]] StructuredBuffer<float> latents;     // [batch][layers - 1][pixels]
[[vk::binding(4, 0)]] RWStructuredBuffer<float> output;    // [batch][outputChannels][pixels]

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 id: SV_DispatchThreadID) {
    if (id.x >= pc.batch * pc.pixels) return;
    const uint n = id.x / pc.pixels, p = id.x % pc.pixels;
    const uint c0 = n * 3u * pc.pixels;
    const TexelQuad quad = texelQuad(pc, coordinates[c0 + p], coordinates[c0 + pc.pixels + p],
                                     coordinates[c0 + 2u * pc.pixels + p], classes[id.x]);
    for (uint channel = 0u; channel < pc.channels; ++channel) {
        float value = 0.0;
        if (quad.face >= 0)
            for (uint layer = 0u; layer < pc.layers; ++layer) {
                const float scale = layer == 0u ? 1.0 : latents[(n * (pc.layers - 1u) + layer - 1u) * pc.pixels + p];
                const uint plane = planeOf(pc, uint(quad.face), layer, channel);
                float sample = 0.0;
                for (uint t = 0u; t < 4u; ++t)
                    sample += quad.weight[t] * table[plane + quad.index[t]];
                value += scale * sample;
            }
        const uint o = (n * pc.outputChannels + pc.outputOffset + channel) * pc.pixels + p;
        output[o] = pc.accumulate != 0u ? output[o] + value : value;
    }
}
