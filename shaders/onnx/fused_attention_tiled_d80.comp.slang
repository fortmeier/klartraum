// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// SD1.5 UNet attention at the second latent resolution: 8 heads of width 80.
// Apple GPUs allow 32 KiB of threadgroup memory per workgroup, and Metal places
// the per-invocation query and accumulator arrays there as well (20 KiB at this
// width). 16-key tiles keep the total at 30 KiB.
import fused_attention_tiled;

static const uint HEAD_DEPTH = 80;
static const uint KEY_TILE = 16;

[[vk::push_constant]] ConstantBuffer<AttentionPushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> queryData;
[[vk::binding(1, 0)]] StructuredBuffer<float> keyData;
[[vk::binding(2, 0)]] StructuredBuffer<float> valueData;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> outputData;

// The staged key tile, [KEY_TILE][HEAD_DEPTH].
groupshared float sharedKey[KEY_TILE * HEAD_DEPTH];
// The staged value tile, [KEY_TILE][HEAD_DEPTH].
groupshared float sharedValue[KEY_TILE * HEAD_DEPTH];

struct Tiles : IAttentionTiles {
    static float key(uint index) { return sharedKey[index]; }
    static void setKey(uint index, float value) { sharedKey[index] = value; }
    static float value(uint index) { return sharedValue[index]; }
    static void setValue(uint index, float value) { sharedValue[index] = value; }
};

[shader("compute")]
[numthreads(QUERIES_PER_GROUP, 1, 1)]
void main(uint3 localId: SV_GroupThreadID, uint3 groupId: SV_GroupID) {
    tiledAttention<HEAD_DEPTH, KEY_TILE, Tiles>(localId.x, groupId, pc, queryData, keyData, valueData, outputData);
}
