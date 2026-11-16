// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// The photo enhancer's photometric loss of an image against its target: l1Weight x the mean absolute difference plus
// colourWeight x the mean absolute difference of their 8 x 8 block means (train_photo_enhancer.py's colour loss).
// Writes the image's gradient and every element's share of the loss (the block term spread evenly over its block), so
// that a sum of the shares is the loss. One invocation per element.

struct PushConstants {
    uint planes; // batch x channels
    uint height;
    uint width;
    float l1Weight;
    float colourWeight;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> image;  // [planes][height][width]
[[vk::binding(1, 0)]] StructuredBuffer<float> target; // [planes][height][width]
[[vk::binding(2, 0)]] RWStructuredBuffer<float> gradient;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> loss;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 id: SV_DispatchThreadID) {
    const uint count = pc.planes * pc.height * pc.width;
    if (id.x >= count) return;
    const uint x = id.x % pc.width, y = (id.x / pc.width) % pc.height, plane = id.x / (pc.width * pc.height);
    const float d = image[id.x] - target[id.x];
    float g = pc.l1Weight * sign(d) / float(count);
    float share = pc.l1Weight * abs(d) / float(count);
    if (pc.colourWeight > 0.0) {
        // The 8 x 8 block of this pixel (avg_pool2d drops incomplete blocks at the border).
        const uint blocksY = pc.height / 8u, blocksX = pc.width / 8u;
        const uint by = y / 8u, bx = x / 8u;
        if (by < blocksY && bx < blocksX) {
            float sum = 0.0;
            for (uint dy = 0u; dy < 8u; ++dy)
                for (uint dx = 0u; dx < 8u; ++dx) {
                    const uint i = (plane * pc.height + by * 8u + dy) * pc.width + bx * 8u + dx;
                    sum += image[i] - target[i];
                }
            const float blocks = float(pc.planes * blocksY * blocksX);
            g += pc.colourWeight * sign(sum) / (blocks * 64.0);
            share += pc.colourWeight * abs(sum / 64.0) / (blocks * 64.0);
        }
    }
    gradient[id.x] = g;
    loss[id.x] = share;
}
