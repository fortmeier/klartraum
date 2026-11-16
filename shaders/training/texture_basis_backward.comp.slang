// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// The backward pass of the code-mixed feature texture (texture_basis.comp.slang): every textured pixel adds its
// features' gradients, times its latent (1 for the base layer) and each texel's bilinear weight, to the four texels of
// every layer; pixels share texels, so the additions are atomic (VK_EXT_shader_atomic_float) and the table's gradient
// must start at zero. The latents' gradient (the features' gradient dotted with each further layer's sample) is
// written per pixel, or added with accumulate. One invocation per pixel.
import texture_basis;

[[vk::push_constant]] ConstantBuffer<TextureBasisConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> coordinates;      // [batch][3][pixels]
[[vk::binding(1, 0)]] StructuredBuffer<float> classes;          // [batch][pixels]
[[vk::binding(2, 0)]] StructuredBuffer<float> table;            // [faces][layers][channels][resolution][resolution]
[[vk::binding(3, 0)]] StructuredBuffer<float> latents;          // [batch][layers - 1][pixels]
[[vk::binding(4, 0)]] StructuredBuffer<float> outputGradient;   // [batch][outputChannels][pixels]
[[vk::binding(5, 0)]] RWStructuredBuffer<float> tableGradient;  // the table's layout
[[vk::binding(6, 0)]] RWStructuredBuffer<float> latentGradient; // [batch][layers - 1][pixels]

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 id: SV_DispatchThreadID) {
    if (id.x >= pc.batch * pc.pixels) return;
    const uint n = id.x / pc.pixels, p = id.x % pc.pixels;
    const uint c0 = n * 3u * pc.pixels;
    const TexelQuad quad = texelQuad(pc, coordinates[c0 + p], coordinates[c0 + pc.pixels + p],
                                     coordinates[c0 + 2u * pc.pixels + p], classes[id.x]);
    for (uint layer = 1u; layer < pc.layers; ++layer) {
        float dLatent = 0.0;
        if (quad.face >= 0)
            for (uint channel = 0u; channel < pc.channels; ++channel) {
                const float g = outputGradient[(n * pc.outputChannels + pc.outputOffset + channel) * pc.pixels + p];
                const uint plane = planeOf(pc, uint(quad.face), layer, channel);
                for (uint t = 0u; t < 4u; ++t)
                    dLatent += g * quad.weight[t] * table[plane + quad.index[t]];
            }
        const uint o = (n * (pc.layers - 1u) + layer - 1u) * pc.pixels + p;
        latentGradient[o] = pc.accumulate != 0u ? latentGradient[o] + dLatent : dLatent;
    }
    if (quad.face < 0) return;
    for (uint layer = 0u; layer < pc.layers; ++layer) {
        const float scale = layer == 0u ? 1.0 : latents[(n * (pc.layers - 1u) + layer - 1u) * pc.pixels + p];
        for (uint channel = 0u; channel < pc.channels; ++channel) {
            const float g = scale * outputGradient[(n * pc.outputChannels + pc.outputOffset + channel) * pc.pixels + p];
            const uint plane = planeOf(pc, uint(quad.face), layer, channel);
            for (uint t = 0u; t < 4u; ++t) {
                float previous;
                InterlockedAdd(tableGradient[plane + quad.index[t]], quad.weight[t] * g, previous);
            }
        }
    }
}
