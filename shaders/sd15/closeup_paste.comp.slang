// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// The generated close-up pasted into the close-up scene through the feathered
// body mask (pasteGenerated() on the GPU). Output per pixel, packed as bytes
// B, G, R, alpha: the pasted colour and the feathered mask.

struct PushConstants {
    uint size; // the close-up is size x size pixels
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> generated; // [3, size, size] in [-1, 1]
[[vk::binding(1, 0)]] [[vk::image_format("rgba8")]] RWTexture2D<float4> scene;
[[vk::binding(2, 0)]] [[vk::image_format("rgba8")]] RWTexture2D<float4> bodyMask;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> closeUp;

uint byteOf(float unorm) { return uint(unorm * 255.0 + 0.5); }

// Coverage of the 5 x 5 box around a pixel, clamped at the borders (featherMask()).
float feather(int2 pixel, int size) {
    float coverage = 0.0;
    for (int dy = -2; dy <= 2; ++dy) {
        for (int dx = -2; dx <= 2; ++dx) {
            int2 p = clamp(pixel + int2(dx, dy), int2(0), int2(size - 1));
            coverage += bodyMask.Load(p).r > 0.0 ? 1.0 : 0.0;
        }
    }
    return coverage / 25.0;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    int2 pixel = int2(globalId.xy);
    int size = int(pc.size);
    if (pixel.x >= size || pixel.y >= size) {
        return;
    }
    uint plane = pc.size * pc.size, i = uint(pixel.y) * pc.size + uint(pixel.x);
    float alpha = feather(pixel, size);
    float4 original = scene.Load(pixel);
    uint channels[3];
    for (int c = 0; c < 3; ++c) {
        float value = (clamp(generated[uint(c) * plane + i], -1.0, 1.0) + 1.0) * 127.5;
        float originalByte = float(byteOf(original[c]));
        channels[c] = uint(alpha * value + (1.0 - alpha) * originalByte + 0.5);
    }
    uint alphaByte = uint(alpha * 255.0 + 0.5);
    closeUp[i] = channels[2] | (channels[1] << 8) | (channels[0] << 16) | (alphaByte << 24);
}
