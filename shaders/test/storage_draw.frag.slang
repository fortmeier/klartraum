// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Test shader of StorageDrawComponent: the draw's colour from its push constants.

struct PushConstants {
    float4 color;
    uint first;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[shader("fragment")]
float4 main() : SV_Target {
    return pc.color;
}
