// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// DiffAugment of the discriminator's pairs (train_photo_enhancer.augment_pair()) with given random numbers, per sample:
// brightness and contrast on the image (the last three channels: + brightness, then the distance to the pixel's mean
// over them times contrast), a roll of all channels by (dy, dx) and a cutout, zero over [cy - H/4, cy + H/4) x
// [cx - W/4, cx + W/4). One invocation per element.

struct PushConstants {
    uint batch;
    uint channels;
    uint height;
    uint width;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> pairs;      // [batch][channels][height][width]
[[vk::binding(1, 0)]] StructuredBuffer<float> parameters; // [batch][6]: brightness, contrast, dy, dx, cy, cx
[[vk::binding(2, 0)]] RWStructuredBuffer<float> augmented;

int wrap(int v, int size) { return ((v % size) + size) % size; }

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 id: SV_DispatchThreadID) {
    const uint pixels = pc.height * pc.width;
    if (id.x >= pc.batch * pc.channels * pixels) return;
    const uint x = id.x % pc.width, y = (id.x / pc.width) % pc.height;
    const uint c = (id.x / pixels) % pc.channels, n = id.x / (pixels * pc.channels);
    const float brightness = parameters[n * 6u], contrast = parameters[n * 6u + 1u];
    const int dy = int(parameters[n * 6u + 2u]), dx = int(parameters[n * 6u + 3u]);
    const int cy = int(parameters[n * 6u + 4u]), cx = int(parameters[n * 6u + 5u]);
    const int qh = int(pc.height / 4u), qw = int(pc.width / 4u);
    if (int(y) >= max(0, cy - qh) && int(y) < cy + qh && int(x) >= max(0, cx - qw) && int(x) < cx + qw) {
        augmented[id.x] = 0.0;
        return;
    }
    // torch.roll: the output at y takes the input at y - dy.
    const uint sy = uint(wrap(int(y) - dy, int(pc.height))), sx = uint(wrap(int(x) - dx, int(pc.width)));
    const uint source = sy * pc.width + sx;
    const uint base = n * pc.channels * pixels;
    float v = pairs[base + c * pixels + source];
    if (c + 3u >= pc.channels) {
        float mean = 0.0;
        for (uint k = pc.channels - 3u; k < pc.channels; ++k) mean += pairs[base + k * pixels + source] + brightness;
        mean /= 3.0;
        v = (v + brightness - mean) * contrast + mean;
    }
    augmented[id.x] = v;
}
