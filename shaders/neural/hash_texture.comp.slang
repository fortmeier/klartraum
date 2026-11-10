// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// A learnable surface texture's features per pixel (scripts/neural/hash_texture.py): the multi-resolution 3D hash grid
// of the pixel's surface class at its coordinates (u, v, object id). On each level the 8 corners of the point's cell
// are looked up -- by direct index where the level's grid of (resolution + 1)^3 corners fits the table, else by the
// spatial hash x * 1 xor y * 2654435761 xor z * 805459861 modulo the table size (a power of two, so the low bits of the
// 32-bit products are those of PyTorch's 64-bit ones) -- and blended trilinearly. Pixels of classes without a grid get
// zeros. One invocation per pixel.

struct PushConstants {
    uint width;
    uint height;
    uint levels;
    uint features;
    uint tableSize;
    uint grids;
    uint outputOffset; // channels of the output before the features
    uint classes[4];
    uint resolutions[16];
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> coordinates; // [3][height][width]
[[vk::binding(1, 0)]] StructuredBuffer<float> classes;     // [height][width]
[[vk::binding(2, 0)]] StructuredBuffer<float> table;       // [grids][levels][tableSize][features]
[[vk::binding(3, 0)]] RWStructuredBuffer<float> output;    // [outputOffset + levels * features][height][width]

uint cornerIndex(uint3 corner, uint resolution) {
    uint r = resolution + 1u;
    // r^3 fits 32 bits for r <= 1290; larger grids never fit a table of at most 2^31 entries.
    if (r <= 1290u && r * r * r <= pc.tableSize)
        return corner.x + r * (corner.y + r * corner.z);
    return ((corner.x * 1u) ^ (corner.y * 2654435761u) ^ (corner.z * 805459861u)) & (pc.tableSize - 1u);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 id: SV_DispatchThreadID) {
    if (id.x >= pc.width || id.y >= pc.height) return;
    const uint pixels = pc.width * pc.height;
    const uint p = id.y * pc.width + id.x;
    const uint pixelClass = uint(classes[p] + 0.5);
    uint grid = 0xffffffffu;
    for (uint g = 0u; g < pc.grids; ++g)
        if (pc.classes[g] == pixelClass) grid = g;
    const float3 point = float3(coordinates[p], coordinates[pixels + p], coordinates[2u * pixels + p]);
    for (uint level = 0u; level < pc.levels; ++level) {
        float sums[4] = { 0.0, 0.0, 0.0, 0.0 };
        if (grid != 0xffffffffu) {
            const uint resolution = pc.resolutions[level];
            const float3 scaled = point * float(resolution);
            const uint3 cell = uint3(clamp(int3(floor(scaled)), int3(0), int3(int(resolution) - 1)));
            const float3 weight = scaled - float3(cell);
            const uint base = (grid * pc.levels + level) * pc.tableSize;
            for (uint bits = 0u; bits < 8u; ++bits) {
                const uint3 offset = uint3(bits & 1u, (bits >> 1u) & 1u, (bits >> 2u) & 1u);
                const float3 w3 = select(offset != uint3(0u), weight, 1.0 - weight);
                const float w = w3.x * w3.y * w3.z;
                const uint entry = (base + cornerIndex(cell + offset, resolution)) * pc.features;
                for (uint f = 0u; f < pc.features; ++f) sums[f] += w * table[entry + f];
            }
        }
        for (uint f = 0u; f < pc.features; ++f)
            output[(pc.outputOffset + level * pc.features + f) * pixels + p] = sums[f];
    }
}
