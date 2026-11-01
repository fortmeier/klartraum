// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// 3D convolution in 256 x 32 tiles with split-K, for few output positions (see conv3d_tiled.slang).
import conv3d_tiled;

static const uint TILE_M = 256;
static const uint TILE_N = 32;

// The weight tile: [k][channel / 4].
groupshared float4 weightTile[TILE_K][TILE_M / 4];
// The gathered input tile: [k][column / 4].
groupshared float4 inputTile[TILE_K][TILE_N / 4];

struct Tiles : IConvTiles {
    static void setWeight(uint k, uint channel, float value) { weightTile[k][channel / 4][channel % 4] = value; }
    static float4 weight4(uint k, uint channelGroup) { return weightTile[k][channelGroup]; }
    static void setInput(uint k, uint column, float value) { inputTile[k][column / 4][column % 4] = value; }
    static float4 input4(uint k, uint columnGroup) { return inputTile[k][columnGroup]; }
};

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint localIndex: SV_GroupIndex, uint3 groupId: SV_GroupID) {
    conv3d<TILE_M, TILE_N, true, Tiles>(localIndex, groupId);
}
