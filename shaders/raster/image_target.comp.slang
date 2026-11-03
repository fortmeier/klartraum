// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Turns a rendered picture (8 bits per channel, copied from an image into a buffer, one uint per pixel) into one
// view of training targets: four floats per pixel, the rgb as drawn (premultiplied over transparent black) and the
// alpha (coverage) as the mask.

struct PushConstants {
    uint pixels;
    uint view;
    uint bgra; // 1: the bytes are b, g, r, a
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<uint> picture;      // per pixel
[[vk::binding(1, 0)]] RWStructuredBuffer<float4> targets;  // per pixel and view

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    const uint pixel = globalId.x;
    if (pixel >= pc.pixels) return;
    const uint word = picture[pixel];
    float4 value = float4(float(word & 0xffu), float((word >> 8u) & 0xffu), float((word >> 16u) & 0xffu),
                          float(word >> 24u)) / 255.0;
    if (pc.bgra != 0u) value = value.bgra;
    targets[pc.view * pc.pixels + pixel] = value;
}
