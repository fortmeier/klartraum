// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// A 2 x 2 max-pool with stride 2 (F.max_pool2d(x, 2)): every output the largest of its window (the first of equal
// ones, in row order), and which of the four it was (0..3, as a float) for the backward pass. One invocation per
// output.

struct PushConstants {
    uint planes; // batch x channels
    uint inputHeight;
    uint inputWidth;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> x;       // [planes][inputHeight][inputWidth]
[[vk::binding(1, 0)]] RWStructuredBuffer<float> y;      // [planes][inputHeight / 2][inputWidth / 2]
[[vk::binding(2, 0)]] RWStructuredBuffer<float> argmax; // as y

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 id: SV_DispatchThreadID) {
    const uint outputHeight = pc.inputHeight / 2u, outputWidth = pc.inputWidth / 2u;
    if (id.x >= pc.planes * outputHeight * outputWidth) return;
    const uint ox = id.x % outputWidth, oy = (id.x / outputWidth) % outputHeight;
    const uint plane = id.x / (outputWidth * outputHeight);
    const uint base = (plane * pc.inputHeight + 2u * oy) * pc.inputWidth + 2u * ox;
    float best = x[base];
    uint which = 0u;
    for (uint k = 1u; k < 4u; ++k) {
        const float v = x[base + (k / 2u) * pc.inputWidth + (k % 2u)];
        if (v > best) {
            best = v;
            which = k;
        }
    }
    y[id.x] = best;
    argmax[id.x] = float(which);
}
