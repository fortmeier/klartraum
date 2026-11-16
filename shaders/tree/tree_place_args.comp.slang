// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import glsl;

// After tree_place_cull.comp: the indirect draws of the visible trees as
// proxies and as impostors (a card each), and the dispatch of their Gaussians
// (64 trees per workgroup).

[[vk::binding(0, 0)]] StructuredBuffer<uint> counter;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> drawArgs;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> dispatchArgs;

struct Push {
    uint capacity;
    uint vertices;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

[shader("compute")]
[numthreads(1, 1, 1)]
void main() {
    uint count = min(counter[0], pc.capacity);
    drawArgs[0] = pc.vertices;
    drawArgs[1] = count;
    drawArgs[2] = 0u;
    drawArgs[3] = 0u;
    drawArgs[4] = 6u;
    drawArgs[5] = count;
    drawArgs[6] = 0u;
    drawArgs[7] = 0u;
    dispatchArgs[0] = (count + 63u) / 64u;
    dispatchArgs[1] = 1u;
    dispatchArgs[2] = 1u;
}
