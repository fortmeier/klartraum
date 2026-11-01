// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

struct Params {
    float4 color;
    float depth;
};
[[vk::push_constant]] ConstantBuffer<Params> pc;

struct VertexOutput {
    float4 position : SV_Position;
    [[vk::location(0)]] float4 color;
};

// One triangle covering the whole viewport at the depth of the push constants.
[shader("vertex")]
VertexOutput main(uint vertexIndex: SV_VulkanVertexID) {
    const float2 positions[3] = { float2(-1, -1), float2(3, -1), float2(-1, 3) };
    VertexOutput output;
    output.position = float4(positions[vertexIndex], pc.depth, 1);
    output.color = pc.color;
    return output;
}
