// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// The input gradient of augment_pair.comp.slang: input pixel (sy, sx) went to output pixel ((sy + dy), (sx + dx))
// (wrapped), whose gradient it takes unless the cutout zeroed it; through the image channels' brightness and
// contrast, d in_j = contrast d out_j + (1 - contrast) / 3 sum_c d out_c. One invocation per input element.

struct PushConstants {
    uint batch;
    uint channels;
    uint height;
    uint width;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> parameters;       // [batch][6]: brightness, contrast, dy, dx, cy, cx
[[vk::binding(1, 0)]] StructuredBuffer<float> outputGradient;   // [batch][channels][height][width]
[[vk::binding(2, 0)]] RWStructuredBuffer<float> inputGradient;  // [batch][channels][height][width]

int wrap(int v, int size) { return ((v % size) + size) % size; }

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 id: SV_DispatchThreadID) {
    const uint pixels = pc.height * pc.width;
    if (id.x >= pc.batch * pc.channels * pixels) return;
    const uint sx = id.x % pc.width, sy = (id.x / pc.width) % pc.height;
    const uint c = (id.x / pixels) % pc.channels, n = id.x / (pixels * pc.channels);
    const float contrast = parameters[n * 6u + 1u];
    const int dy = int(parameters[n * 6u + 2u]), dx = int(parameters[n * 6u + 3u]);
    const int cy = int(parameters[n * 6u + 4u]), cx = int(parameters[n * 6u + 5u]);
    const int qh = int(pc.height / 4u), qw = int(pc.width / 4u);
    const uint y = uint(wrap(int(sy) + dy, int(pc.height))), x = uint(wrap(int(sx) + dx, int(pc.width)));
    if (int(y) >= max(0, cy - qh) && int(y) < cy + qh && int(x) >= max(0, cx - qw) && int(x) < cx + qw) {
        inputGradient[id.x] = 0.0;
        return;
    }
    const uint base = n * pc.channels * pixels, at = y * pc.width + x;
    float g = outputGradient[base + c * pixels + at];
    if (c + 3u >= pc.channels) {
        float sum = 0.0;
        for (uint k = pc.channels - 3u; k < pc.channels; ++k) sum += outputGradient[base + k * pixels + at];
        g = contrast * g + (1.0 - contrast) / 3.0 * sum;
    }
    inputGradient[id.x] = g;
}
