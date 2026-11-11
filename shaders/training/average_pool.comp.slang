// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// A 2 x 2 average pool with stride 2 (F.avg_pool2d(x, 2)). One invocation per output.

struct PushConstants {
    uint planes;
    uint inputHeight;
    uint inputWidth;
    uint accumulate;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> x;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> y;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 id: SV_DispatchThreadID) {
    const uint oh = pc.inputHeight / 2u, ow = pc.inputWidth / 2u;
    if (id.x >= pc.planes * oh * ow) return;
    const uint ox = id.x % ow, oy = (id.x / ow) % oh, plane = id.x / (ow * oh);
    const uint base = (plane * pc.inputHeight + 2u * oy) * pc.inputWidth + 2u * ox;
    y[id.x] = 0.25 * (x[base] + x[base + 1u] + x[base + pc.inputWidth] + x[base + pc.inputWidth + 1u]);
}
