// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

struct PushConstants {
    // Screen size in pixels, used to map pixel-space positions to NDC.
    float2 screenSize;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pushConstants;

struct VertexOutput {
    float4 position : SV_Position;
    [[vk::location(0)]] float2 fragUV;
    [[vk::location(1)]] float4 fragColor;
};

[shader("vertex")]
VertexOutput main([[vk::location(0)]] float2 inPosition, [[vk::location(1)]] float2 inUV,
                  [[vk::location(2)]] float4 inColor) {
    // Pixel-space (origin top-left, y down) maps directly to Vulkan NDC
    // (origin top-left, y down) by normalizing to [0, 2] and shifting by -1.
    float2 ndc = (inPosition / pushConstants.screenSize) * 2.0 - 1.0;
    VertexOutput output;
    output.position = float4(ndc, 0.0, 1.0);
    output.fragUV = inUV;
    output.fragColor = inColor;
    return output;
}
