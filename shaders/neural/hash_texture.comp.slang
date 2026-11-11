// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// A learnable surface texture's features per pixel (scripts/neural/hash_texture.py): the multi-resolution 3D hash grid
// of the pixel's surface class at its coordinates (u, v, object id), looked up and blended as hash_texture.slang
// describes. Pixels of classes without a grid get zeros. One invocation per pixel; z the image of the batch.
import hash_texture;

[[vk::push_constant]] ConstantBuffer<HashTextureConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> coordinates; // [batch][3][height][width]
[[vk::binding(1, 0)]] StructuredBuffer<float> classes;     // [batch][height][width]
[[vk::binding(2, 0)]] StructuredBuffer<float> table;       // [grids][levels][tableSize][features]
[[vk::binding(3, 0)]] RWStructuredBuffer<float> output;    // [batch][outputOffset + levels * features][height][width]

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 id: SV_DispatchThreadID) {
    if (id.x >= pc.width || id.y >= pc.height || id.z >= pc.batch) return;
    const uint pixels = pc.width * pc.height;
    const uint p = id.y * pc.width + id.x;
    const uint image = id.z;
    const uint grid = gridOf(pc, classes[image * pixels + p]);
    const uint c0 = image * 3u * pixels;
    const float3 point = float3(coordinates[c0 + p], coordinates[c0 + pixels + p], coordinates[c0 + 2u * pixels + p]);
    const uint outputBase = image * (pc.outputOffset + pc.levels * pc.features) * pixels;
    for (uint level = 0u; level < pc.levels; ++level) {
        float sums[4] = { 0.0, 0.0, 0.0, 0.0 };
        if (grid != 0xffffffffu) {
            for (uint bits = 0u; bits < 8u; ++bits) {
                uint entry;
                float w;
                corner(pc, point, grid, level, bits, entry, w);
                for (uint f = 0u; f < pc.features; ++f) sums[f] += w * table[entry + f];
            }
        }
        for (uint f = 0u; f < pc.features; ++f)
            output[outputBase + (pc.outputOffset + level * pc.features + f) * pixels + p] = sums[f];
    }
}
