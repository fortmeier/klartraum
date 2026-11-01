// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

[shader("fragment")]
float4 main([[vk::location(0)]] float3 fragColor) : SV_Target0 {
    return float4(fragColor, 1.0);
}
