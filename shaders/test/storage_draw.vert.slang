// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Test shader of StorageDrawComponent: vertex i of a draw is position pc.first + i of the storage buffer (clip
// space), in the draw's colour.

struct PushConstants {
    float4 color;
    uint first;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 1)]] StructuredBuffer<float4> positions;

[shader("vertex")]
float4 main(uint vertex: SV_VertexID) : SV_Position {
    return positions[pc.first + vertex];
}
