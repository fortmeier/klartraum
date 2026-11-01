// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

[shader("fragment")]
float4 main([[vk::location(0)]] float4 color) : SV_Target0 {
    return color;
}
