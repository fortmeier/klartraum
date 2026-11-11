// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// The table gradient of the surface texture (hash_texture.comp.slang): every pixel of a class with a grid adds its
// features' gradients, times each corner's trilinear weight, to the 8 corner entries of its cell on every level.
// Pixels share entries, so the additions are atomic (VK_EXT_shader_atomic_float); the gradient must start at zero.
import hash_texture;

[[vk::push_constant]] ConstantBuffer<HashTextureConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> coordinates;    // [batch][3][height][width]
[[vk::binding(1, 0)]] StructuredBuffer<float> classes;        // [batch][height][width]
[[vk::binding(2, 0)]] StructuredBuffer<float> outputGradient; // the output's layout
[[vk::binding(3, 0)]] RWStructuredBuffer<float> tableGradient; // [grids][levels][tableSize][features]

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 id: SV_DispatchThreadID) {
    if (id.x >= pc.width || id.y >= pc.height || id.z >= pc.batch) return;
    const uint pixels = pc.width * pc.height;
    const uint p = id.y * pc.width + id.x;
    const uint image = id.z;
    const uint grid = gridOf(pc, classes[image * pixels + p]);
    if (grid == 0xffffffffu) return;
    const uint c0 = image * 3u * pixels;
    const float3 point = float3(coordinates[c0 + p], coordinates[c0 + pixels + p], coordinates[c0 + 2u * pixels + p]);
    const uint gradientBase = image * (pc.outputOffset + pc.levels * pc.features) * pixels;
    for (uint level = 0u; level < pc.levels; ++level) {
        float gradient[4];
        for (uint f = 0u; f < pc.features; ++f)
            gradient[f] = outputGradient[gradientBase + (pc.outputOffset + level * pc.features + f) * pixels + p];
        for (uint bits = 0u; bits < 8u; ++bits) {
            uint entry;
            float w;
            corner(pc, point, grid, level, bits, entry, w);
            for (uint f = 0u; f < pc.features; ++f) {
                float previous;
                InterlockedAdd(tableGradient[entry + f], w * gradient[f], previous);
            }
        }
    }
}
