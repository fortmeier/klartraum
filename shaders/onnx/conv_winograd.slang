// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// 3x3 convolution with stride 1 and padding 1 as Winograd F(2x2, 3x3)
// (Lavin & Gray 2016): every 2x2 output tile takes 16 products instead of 36.
// Three passes on one dispatch grid, the batched product's:
//  1. conv_winograd_transform: U = G g Gᵀ for every (output, input) channel
//     pair and V = Bᵀ d B for every input channel and 4x4 input tile d;
//  2. conv_winograd_gemm: M[e] = U[e] x V[e] for the 16 tile positions e,
//     with the MatMul kernel;
//  3. conv_winograd_output: Y = Aᵀ M A plus the bias, for every output
//     channel and tile.
// The transforms' integer coefficients and halves are exact in float32; only
// the order of the additions differs from a direct convolution.
// Layouts: U [16][outputChannels][inputChannels], V [16][inputChannels][tiles],
// M [16][outputChannels][tiles], tiles = batch x tilesY x tilesX.
module conv_winograd;

public struct WinogradPushConstants {
    public uint batch;
    public uint inputChannels;
    public uint outputChannels;
    public uint height; // the input's, which is the output's
    public uint width;
    public uint tilesY; // ceil(height / 2)
    public uint tilesX;
};
[[vk::push_constant]] public ConstantBuffer<WinogradPushConstants> pc;

// The index of this invocation in the whole dispatch and the dispatch's invocation count, for grid-stride loops.
public uint2 gridPosition(uint localIndex, uint3 groupId) {
    const uint3 groups = WorkgroupCount();
    const uint group = (groupId.z * groups.y + groupId.y) * groups.x + groupId.x;
    return uint2(group * 256 + localIndex, groups.x * groups.y * groups.z * 256);
}

public uint tileCount() { return pc.batch * pc.tilesY * pc.tilesX; }

public void transformWeights(uint item, StructuredBuffer<float> weights, RWStructuredBuffer<float> u) {
    // item = output channel x input channel
    const uint pairs = pc.outputChannels * pc.inputChannels;
    float g[3][3];
    for (uint r = 0; r < 3; ++r)
        for (uint c = 0; c < 3; ++c) g[r][c] = weights[item * 9 + r * 3 + c];
    // G g: rows (g0, (g0 + g1 + g2) / 2, (g0 - g1 + g2) / 2, g2), per column.
    float t[4][3];
    for (uint c = 0; c < 3; ++c) {
        t[0][c] = g[0][c];
        t[1][c] = 0.5 * (g[0][c] + g[1][c] + g[2][c]);
        t[2][c] = 0.5 * (g[0][c] - g[1][c] + g[2][c]);
        t[3][c] = g[2][c];
    }
    for (uint r = 0; r < 4; ++r) {
        const float values[4] = { t[r][0], 0.5 * (t[r][0] + t[r][1] + t[r][2]), 0.5 * (t[r][0] - t[r][1] + t[r][2]),
                                  t[r][2] };
        for (uint c = 0; c < 4; ++c) u[(r * 4 + c) * pairs + item] = values[c];
    }
}

public void transformInput(uint item, StructuredBuffer<float> inputData, RWStructuredBuffer<float> v) {
    // item = input channel x tile; neighbouring items are neighbouring tiles of one channel.
    const uint tiles = tileCount();
    const uint channel = item / tiles, tile = item % tiles;
    const uint tilesPerImage = pc.tilesY * pc.tilesX;
    const uint batch = tile / tilesPerImage;
    const uint ty = (tile % tilesPerImage) / pc.tilesX, tx = tile % pc.tilesX;
    const uint plane = pc.height * pc.width;
    const uint base = (batch * pc.inputChannels + channel) * plane;
    float d[4][4];
    for (uint r = 0; r < 4; ++r) {
        const int y = int(2 * ty + r) - 1;
        for (uint c = 0; c < 4; ++c) {
            const int x = int(2 * tx + c) - 1;
            d[r][c] = y >= 0 && y < int(pc.height) && x >= 0 && x < int(pc.width) ? inputData[base + uint(y) * pc.width + uint(x)]
                                                                                   : 0.0;
        }
    }
    // Bᵀ d: rows (d0 - d2, d1 + d2, d2 - d1, d1 - d3), per column; then the same on the rows.
    float t[4][4];
    for (uint c = 0; c < 4; ++c) {
        t[0][c] = d[0][c] - d[2][c];
        t[1][c] = d[1][c] + d[2][c];
        t[2][c] = d[2][c] - d[1][c];
        t[3][c] = d[1][c] - d[3][c];
    }
    const uint stride = pc.inputChannels * tiles;
    for (uint r = 0; r < 4; ++r) {
        const float values[4] = { t[r][0] - t[r][2], t[r][1] + t[r][2], t[r][2] - t[r][1], t[r][1] - t[r][3] };
        for (uint c = 0; c < 4; ++c) v[(r * 4 + c) * stride + item] = values[c];
    }
}

public void transformOutput(uint item, StructuredBuffer<float> m, StructuredBuffer<float> bias,
                            RWStructuredBuffer<float> outputData) {
    // item = output channel x tile
    const uint tiles = tileCount();
    const uint channel = item / tiles, tile = item % tiles;
    const uint stride = pc.outputChannels * tiles;
    float p[4][4];
    for (uint e = 0; e < 16; ++e) p[e / 4][e % 4] = m[e * stride + item];
    // Aᵀ p: rows (p0 + p1 + p2, p1 - p2 - p3), per column; then the same on the rows.
    float t[2][4];
    for (uint c = 0; c < 4; ++c) {
        t[0][c] = p[0][c] + p[1][c] + p[2][c];
        t[1][c] = p[1][c] - p[2][c] - p[3][c];
    }
    const uint tilesPerImage = pc.tilesY * pc.tilesX;
    const uint batch = tile / tilesPerImage;
    const uint ty = (tile % tilesPerImage) / pc.tilesX, tx = tile % pc.tilesX;
    const uint plane = pc.height * pc.width;
    const uint base = (batch * pc.outputChannels + channel) * plane;
    const float b = bias[channel];
    for (uint r = 0; r < 2; ++r) {
        const uint y = 2 * ty + r;
        if (y >= pc.height) continue;
        const float values[2] = { t[r][0] + t[r][1] + t[r][2], t[r][1] - t[r][2] - t[r][3] };
        for (uint c = 0; c < 2; ++c) {
            const uint x = 2 * tx + c;
            if (x < pc.width) outputData[base + y * pc.width + x] = values[c] + b;
        }
    }
}
