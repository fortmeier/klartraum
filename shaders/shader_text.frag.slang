// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// The glyph atlas: a single-channel coverage mask, red holds glyph alpha.
[[vk::binding(0, 0)]] Sampler2D glyphAtlas;

[shader("fragment")]
float4 main([[vk::location(0)]] float2 fragUV, [[vk::location(1)]] float4 fragColor) : SV_Target0 {
    float coverage = glyphAtlas.Sample(fragUV).r;
    return float4(fragColor.rgb, fragColor.a * coverage);
}
