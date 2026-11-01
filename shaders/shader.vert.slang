// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

struct UniformBufferObject {
    float4x4 model;
    float4x4 view;
    float4x4 proj;
};
[[vk::binding(0, 0)]] ConstantBuffer<UniformBufferObject> ubo;

struct VertexOutput {
    float4 position : SV_Position;
    [[vk::location(0)]] float3 fragColor;
};

[shader("vertex")]
VertexOutput main([[vk::location(0)]] float3 inPosition, [[vk::location(1)]] float3 inColor) {
    VertexOutput output;
    output.position = mul(mul(mul(ubo.proj, ubo.view), ubo.model), float4(inPosition, 1.0));
    output.fragColor = inColor;
    return output;
}
