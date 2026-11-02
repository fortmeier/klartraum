// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Advances the drawn view of a training graph on the GPU: view = (view + 1) % viewCount.

struct PushConstants {
    uint viewCount;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] RWStructuredBuffer<uint> viewIndex;

[shader("compute")]
[numthreads(1, 1, 1)]
void main() {
    viewIndex[0] = (viewIndex[0] + 1) % pc.viewCount;
}
