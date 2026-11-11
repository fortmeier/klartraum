// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// The lookup of a learnable surface texture (scripts/neural/hash_texture.py), shared by hash_texture.comp.slang and
// its backward pass: on each level the 8 corners of the point's cell, by direct index where the level's grid of
// (resolution + 1)^3 corners fits the table, else by the spatial hash x * 1 xor y * 2654435761 xor z * 805459861
// modulo the table size (a power of two, so the low bits of the 32-bit products are those of PyTorch's 64-bit ones),
// blended trilinearly.
module hash_texture;

public struct HashTextureConstants {
    public uint width;
    public uint height;
    public uint levels;
    public uint features;
    public uint tableSize;
    public uint grids;
    public uint outputOffset; // channels of the output before the features
    public uint batch;
    public uint classes[4];
    public uint resolutions[16];
};

uint cornerIndex(uint3 corner, uint resolution, uint tableSize) {
    uint r = resolution + 1u;
    // r^3 fits 32 bits for r <= 1290; larger grids never fit a table of at most 2^31 entries.
    if (r <= 1290u && r * r * r <= tableSize)
        return corner.x + r * (corner.y + r * corner.z);
    return ((corner.x * 1u) ^ (corner.y * 2654435761u) ^ (corner.z * 805459861u)) & (tableSize - 1u);
}

// The grid of a pixel's class, or 0xffffffff for classes without one.
public uint gridOf(HashTextureConstants pc, float pixelClass) {
    const uint c = uint(pixelClass + 0.5);
    uint grid = 0xffffffffu;
    for (uint g = 0u; g < pc.grids; ++g)
        if (pc.classes[g] == c) grid = g;
    return grid;
}

// Corner `bits` (x, y, z offsets in bits 0, 1, 2) of the point's cell on `level`: the index of its first feature in
// the table and its trilinear weight.
public void corner(HashTextureConstants pc, float3 point, uint grid, uint level, uint bits, out uint entry,
                   out float weight) {
    const uint resolution = pc.resolutions[level];
    const float3 scaled = point * float(resolution);
    const uint3 cell = uint3(clamp(int3(floor(scaled)), int3(0), int3(int(resolution) - 1)));
    const float3 fraction = scaled - float3(cell);
    const uint3 offset = uint3(bits & 1u, (bits >> 1u) & 1u, (bits >> 2u) & 1u);
    const float3 w3 = select(offset != uint3(0u), fraction, 1.0 - fraction);
    weight = w3.x * w3.y * w3.z;
    entry = ((grid * pc.levels + level) * pc.tableSize + cornerIndex(cell + offset, resolution, pc.tableSize)) *
            pc.features;
}
