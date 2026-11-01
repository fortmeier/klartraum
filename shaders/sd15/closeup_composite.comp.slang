// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// The full view with the pasted close-up (compositeCloseUp() on the GPU):
// inside the feathered character mask, the close-up and its alpha are sampled
// bilinearly at the pixel's position in the close-up window and blended over
// the scene; elsewhere the scene stays. Output packed as bytes B, G, R, A.

struct PushConstants {
    uint width; // the full view
    uint height;
    uint size; // the close-up
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] [[vk::image_format("rgba8")]] RWTexture2D<float4> scene;
[[vk::binding(1, 0)]] [[vk::image_format("rgba8")]] RWTexture2D<float4> characterMask;
[[vk::binding(2, 0)]] StructuredBuffer<uint> closeUp; // B, G, R, alpha
[[vk::binding(3, 0)]] StructuredBuffer<float> state;  // centre.xy, halfExtent.xy, ...
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> frame;

uint byteOf(float unorm) { return uint(unorm * 255.0 + 0.5); }

float channelAt(int x, int y, int channel) {
    return float((closeUp[uint(y) * pc.size + uint(x)] >> (8 * channel)) & 0xffu);
}

// Bilinear sample of a byte channel at continuous pixel coordinates (pixel centres at integers).
float sampleChannel(float2 position, int channel) {
    int size = int(pc.size);
    float x = clamp(position.x, 0.0, float(size - 1)), y = clamp(position.y, 0.0, float(size - 1));
    int x0 = int(x), y0 = int(y), x1 = min(x0 + 1, size - 1), y1 = min(y0 + 1, size - 1);
    float fx = x - float(x0), fy = y - float(y0);
    return (1.0 - fy) * ((1.0 - fx) * channelAt(x0, y0, channel) + fx * channelAt(x1, y0, channel)) +
           fy * ((1.0 - fx) * channelAt(x0, y1, channel) + fx * channelAt(x1, y1, channel));
}

bool insideMask(int2 pixel) {
    int2 last = int2(pc.width, pc.height) - 1;
    for (int dy = -2; dy <= 2; ++dy) {
        for (int dx = -2; dx <= 2; ++dx) {
            if (characterMask.Load(clamp(pixel + int2(dx, dy), int2(0), last)).r > 0.0) {
                return true;
            }
        }
    }
    return false;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    int2 pixel = int2(globalId.xy);
    if (pixel.x >= int(pc.width) || pixel.y >= int(pc.height)) {
        return;
    }
    float4 original = scene.Load(pixel);
    uint bytes[4] = { byteOf(original.b), byteOf(original.g), byteOf(original.r), byteOf(original.a) };
    if (insideMask(pixel)) {
        float2 ndc = (float2(pixel) + 0.5) / float2(pc.width, pc.height) * 2.0 - 1.0;
        float2 closeUpNdc = (ndc - float2(state[0], state[1])) / float2(state[2], state[3]);
        float2 source = (closeUpNdc * 0.5 + 0.5) * float(pc.size) - 0.5;
        float alpha = sampleChannel(source, 3) / 255.0;
        for (int c = 0; c < 3; ++c) {
            float value = sampleChannel(source, c);
            bytes[c] = uint(alpha * value + (1.0 - alpha) * float(bytes[c]) + 0.5);
        }
    }
    frame[uint(pixel.y) * pc.width + uint(pixel.x)] = bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24);
}
